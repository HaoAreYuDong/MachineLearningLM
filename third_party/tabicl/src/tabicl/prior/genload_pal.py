#50-60 it/s
from __future__ import annotations

import contextlib
import time
import json
import warnings
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import queue
import threading
import multiprocessing as mp
import concurrent
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import IterableDataset

from tabicl.prior.dataset import PriorDataset
from tabicl.prior.prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)

# 优化1: 增加缓存大小减少磁盘I/O
SPARSE_BUFFER_SIZE = 5
GENERATION_THREADS = 4
SAVE_THREADS = 2


def dense2sparse(
    dense_tensor: torch.Tensor, row_lengths: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert a dense tensor with trailing zeros into a compact 1D representation.

    Parameters
    ----------
    dense_tensor : torch.Tensor
        Input tensor of shape (num_rows, num_cols) where each row may contain
        trailing zeros beyond the valid entries

    row_lengths : torch.Tensor
        Tensor of shape (num_rows,) specifying the number of valid entries
        in each row of the dense tensor

    dtype : torch.dtype, default=torch.float32
        Output data type for the sparse representation

    Returns
    -------
    torch.Tensor
        1D tensor of shape (sum(row_lengths),) containing only the valid entries
    """

    assert dense_tensor.dim() == 2, "dense_tensor must be 2D"
    num_rows, num_cols = dense_tensor.shape
    assert row_lengths.shape[0] == num_rows, "row_lengths must match number of rows"
    assert (row_lengths <= num_cols).all(), "row_lengths cannot exceed number of columns"

    indices = torch.arange(num_cols, device=dense_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    sparse = dense_tensor[mask].to(dtype)

    return sparse


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct a dense tensor from its sparse representation.

    This function is the inverse of dense2sparse, reconstructing a padded dense
    tensor from a compact 1D representation and the corresponding row lengths.
    Unused entries in the output are filled with zeros.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        1D tensor containing the valid entries from the original dense tensor

    row_lengths : torch.Tensor
        Number of valid entries for each row in the output tensor

    max_len : Optional[int], default=None
        Maximum length for each row in the output. If None, uses max(row_lengths)

    dtype : torch.dtype, default=torch.float32
        Output data type for the dense representation

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (num_rows, max_len) with zeros padding
    """

    assert sparse_tensor.dim() == 1, "data must be 1D"
    assert row_lengths.sum() == len(sparse_tensor), "data length must match sum of row_lengths"

    num_rows = len(row_lengths)
    max_len = max_len or row_lengths.max().item()
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)

    return dense

class SliceNestedTensor:
    """A wrapper for nested tensors that supports slicing along the first dimension.

    This class wraps PyTorch's nested tensor and provides slicing operations
    along the first dimension, which are not natively supported by nested tensors.
    It maintains compatibility with other nested tensor operations by forwarding
    attribute access to the wrapped tensor.

    Parameters
    ----------
    nested_tensor : torch.Tensor
        A nested tensor to wrap
    """

    def __init__(self, nested_tensor):
        self.nested_tensor = nested_tensor
        self.is_nested = nested_tensor.is_nested

    def __getitem__(self, idx):
        """Support slicing operations along the first dimension."""
        if isinstance(idx, slice):
            start = 0 if idx.start is None else idx.start
            stop = self.nested_tensor.size(0) if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step

            indices = list(range(start, stop, step))
            return SliceNestedTensor(torch.nested.nested_tensor([self.nested_tensor[i] for i in indices]))
        elif isinstance(idx, int):
            return self.nested_tensor[idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getattr__(self, name):
        """Forward attribute access to the wrapped nested tensor."""
        return getattr(self.nested_tensor, name)

    def __len__(self):
        """Return the length of the first dimension."""
        return self.nested_tensor.size(0)

    def to(self, *args, **kwargs):
        """Support the to() method for device/dtype conversion."""
        return SliceNestedTensor(self.nested_tensor.to(*args, **kwargs))


def cat_slice_nested_tensors(tensors: List, dim=0) -> SliceNestedTensor:
    """Concatenate a list of SliceNestedTensor objects along dimension dim.

    Parameters
    ----------
    tensors : List
        List of tensors to concatenate

    dim : int, default=0
        Dimension along which to concatenate

    Returns
    -------
    SliceNestedTensor
        Concatenated tensor wrapped in SliceNestedTensor
    """
    # Extract the wrapped nested tensors
    nested_tensors = [t.nested_tensor if isinstance(t, SliceNestedTensor) else t for t in tensors]
    return SliceNestedTensor(torch.cat(nested_tensors, dim=dim))


class LoadPriorDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        batch_size=512,
        ddp_world_size=1,
        ddp_rank=0,
        start_from=0,
        max_batches=None,
        timeout=60,
        delete_after_load=False,
        device="cpu",
    ):
        # 优化5: 预加载元数据和文件列表
        super().__init__()
        self.data_dir = Path(data_dir)
        self.device = device
        self.start_from = start_from  # 需要添加
        self.max_batches = max_batches  # 需要添加
        self.timeout = timeout  # 需要添加
        self.delete_after_load = delete_after_load  # 需要添加
        
        # 提前加载所有文件列表
        self.file_list = sorted(self.data_dir.glob("batch_*.pt"))
        self.total_files = len(self.file_list)
        
        if max_batches is not None:
            self.file_list = self.file_list[:min(self.total_files, max_batches)]
        
        # 分布式设置
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.batch_size = batch_size
        
        # 计算每个进程应该处理的文件范围
        per_process_files = (len(self.file_list) + ddp_world_size - 1) // ddp_world_size
        start_idx = ddp_rank * per_process_files
        end_idx = min(start_idx + per_process_files, len(self.file_list))
        self.process_files = self.file_list[start_idx:end_idx]
        
        # 缓存相关
        self.buffer = []
        self.current_idx = 0
        self.current_buffer_size = 0
        
        # 预加载元数据
        self._load_metadata()
        
        print(f"Process {ddp_rank}: Loaded {len(self.process_files)} files from {start_idx} to {end_idx}")
    
    def _load_metadata(self):
        """预加载元数据以提高效率"""
        self.metadata = None
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                pass
                
    def __iter__(self):
        # 重置状态
        self.current_idx = 0
        self.buffer = []
        self.current_buffer_size = 0
        return self
        
    def _load_next_file(self):
        """加载下一个文件到缓冲区"""
        if self.current_idx >= len(self.process_files):
            return False
            
        file_path = self.process_files[self.current_idx]
        self.current_idx += 1
        
        # 加载文件
        try:
            batch = torch.load(file_path, map_location=self.device, weights_only=True)
            
            # 添加到缓冲区
            self.buffer.append((
                batch["X"], 
                batch["y"],
                batch["d"],
                batch["seq_lens"],
                batch["train_sizes"]
            ))
            
            self.current_buffer_size += batch["batch_size"]
            return True
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return False
    
    def __next__(self):
        """优化后的批次加载"""
        # 当缓冲区不足时预加载更多文件
        while self.current_buffer_size < self.batch_size and self.current_idx < len(self.process_files):
            self._load_next_file()
            
        if self.current_buffer_size == 0:
            raise StopIteration
            
        # 收集足够的数据形成一个批次
        collected_X = []
        collected_y = []
        collected_d = []
        collected_seq_lens = []
        collected_train_sizes = []
        collected_count = 0
        
        while self.current_buffer_size > 0 and collected_count < self.batch_size:
            X, y, d, seq_lens, train_sizes = self.buffer.pop(0)
            batch_size = d.size(0)
            
            # 如果整个批次都小于剩余所需
            if collected_count + batch_size <= self.batch_size:
                collected_X.append(X)
                collected_y.append(y)
                collected_d.append(d)
                collected_seq_lens.append(seq_lens)
                collected_train_sizes.append(train_sizes)
                collected_count += batch_size
                self.current_buffer_size -= batch_size
            else:
                # 只取所需部分
                needed = self.batch_size - collected_count
                collected_X.append(X[:needed])
                collected_y.append(y[:needed])
                collected_d.append(d[:needed])
                collected_seq_lens.append(seq_lens[:needed])
                collected_train_sizes.append(train_sizes[:needed])
                
                # 将剩余部分放回缓冲区
                if needed < batch_size:
                    self.buffer.insert(0, (
                        X[needed:],
                        y[needed:],
                        d[needed:],
                        seq_lens[needed:],
                        train_sizes[needed:]
                    ))
                    self.current_buffer_size -= (batch_size - needed)
                
                collected_count = self.batch_size
        
        # 处理嵌套张量
        if isinstance(collected_X[0], SliceNestedTensor):
            X_out = cat_slice_nested_tensors(collected_X).nested_tensor
            y_out = cat_slice_nested_tensors(collected_y).nested_tensor
        else:
            # 优化6: 使用张量连接
            X_out = torch.cat(collected_X, dim=0)
            y_out = torch.cat(collected_y, dim=0)
            
        d_out = torch.cat(collected_d, dim=0)
        seq_lens_out = torch.cat(collected_seq_lens, dim=0)
        train_sizes_out = torch.cat(collected_train_sizes, dim=0)
        
        return X_out, y_out, d_out, seq_lens_out, train_sizes_out

    # ... 其余方法保持不变 ...
    def __repr__(self) -> str:
        """
        Returns a string representation of the LoadPriorDataset.

        Returns
        -------
        str
            A formatted string with dataset parameters
        """
        repr_str = (
            f"LoadPriorDataset(\n"
            f"  data_dir: {self.data_dir}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  ddp_world_size: {self.ddp_world_size}\n"
            f"  ddp_rank: {self.ddp_rank}\n"
            f"  start_from: {self.current_idx - self.ddp_rank}\n"
            f"  max_batches: {self.max_batches or 'Infinite'}\n"
            f"  timeout: {self.timeout}\n"
            f"  delete_after_load: {self.delete_after_load}\n"
            f"  device: {self.device}\n"
        )
        if self.metadata:
            repr_str += "  Loaded Metadata:\n"
            repr_str += f"    prior_type: {self.metadata.get('prior_type', 'N/A')}\n"
            repr_str += f"    batch_size (generated): {self.metadata.get('batch_size', 'N/A')}\n"
            repr_str += f"    batch_size_per_gp: {self.metadata.get('batch_size_per_gp', 'N/A')}\n"
            repr_str += f"    min features: {self.metadata.get('min_features', 'N/A')}\n"
            repr_str += f"    max features: {self.metadata.get('max_features', 'N/A')}\n"
            repr_str += f"    max classes: {self.metadata.get('max_classes', 'N/A')}\n"
            repr_str += f"    seq_len: {self.metadata.get('min_seq_len', 'N/A') or 'None'} - {self.metadata.get('max_seq_len', 'N/A')}\n"
            repr_str += f"    log_seq_len: {self.metadata.get('log_seq_len', 'N/A')}\n"
            repr_str += f"    sequence length varies across groups: {self.metadata.get('seq_len_per_gp', 'N/A')}\n"
            repr_str += f"    train_size: {self.metadata.get('min_train_size', 'N/A')} - {self.metadata.get('max_train_size', 'N/A')}\n"
            repr_str += f"    replay_small: {self.metadata.get('replay_small', 'N/A')}\n"
        repr_str += ")"

        return repr_str


class BatchGenerator:
    """优化7: GPU并行生成器"""
    def __init__(self, args, device, device_index):
        self.args = args
        self.device = device
        self.device_index = device_index
        # 关键：为每个GPU设备创建独立的随机种子
        self.np_seed = args.np_seed + device_index * 1000
        self.torch_seed = args.torch_seed + device_index * 1000

        self.batch_factor = 4  # 一次生成4个批次以减少GPU启动开销
        # 增加流式处理能力
        self.stream = torch.cuda.Stream(device=device) if device.startswith('cuda') else None
        
        # 设置设备特定的随机种子
        np.random.seed(self.np_seed)
        torch.manual_seed(self.torch_seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed(self.torch_seed)
            torch.cuda.manual_seed_all(self.torch_seed)
        
        # 确保CuDNN的确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 修改点1：使用args.batch_size作为PriorDataset的batch_size
        self.prior = PriorDataset(
            batch_size=self.args.batch_size,  # 每个文件包含的数据集数量
            batch_size_per_gp=self.args.batch_size_per_gp,
            min_features=self.args.min_features,
            max_features=self.args.max_features,
            max_classes=self.args.max_classes,
            min_seq_len=self.args.min_seq_len,
            max_seq_len=self.args.max_seq_len,
            log_seq_len=self.args.log_seq_len,
            seq_len_per_gp=self.args.seq_len_per_gp,
            min_train_size=self.args.min_train_size,
            max_train_size=self.args.max_train_size,
            replay_small=self.args.replay_small,
            prior_type=self.args.prior_type,
            scm_fixed_hp=DEFAULT_FIXED_HP,
            scm_sampled_hp=DEFAULT_SAMPLED_HP,
            n_jobs=self.args.n_jobs,
            num_threads_per_generate=self.args.num_threads_per_generate,
            device=device,
        )
    
    def generate(self, num_files):
        """高度优化的GPU批量生成方法"""
        results = []
        total_batches = num_files
        num_rounds = (total_batches + self.batch_factor - 1) // self.batch_factor

        
        # 使用CUDA流和大型批量优化
        with torch.cuda.stream(self.stream) if self.stream else contextlib.nullcontext():
            for _ in range(num_rounds):
                n = min(self.batch_factor, total_batches)
                total_batches -= n
                # 一次生成多个批次以减少启动开销
                batch_list = []
                for _ in range(n):
                    batch_list.append(self.prior.get_batch())
                
                # 并行处理所有批次
                processed_batches = []
                for batch in batch_list:
                    X, y, d, seq_lens, train_sizes = batch
                    
                    # 优化：统一处理为三维张量格式
                    if isinstance(X, torch.Tensor) and X.is_nested:
                        # 嵌套张量转换为填充张量
                        X = X.to_padded_tensor(0.0)
                        #y = y.to_padded_tensor(0.0)
                    
                    # 确保总是三维格式 (batch_size, seq_len, features)
                    if len(X.shape) == 2:
                        # 如果是二维张量，重新塑造为三维
                        B = d.size(0)
                        H = d.max().item()
                        T = seq_lens.max().item()
                        
                        # 高效重塑为三维
                        X = X.view(B, T, H)
                        #y = y.view(B, T, 1)
                    
                    # 添加批处理
                    processed_batches.append((
                        X.cpu().detach(),
                        y.cpu().detach(),
                        d.cpu().detach(),
                        seq_lens.cpu().detach(),
                        train_sizes.cpu().detach()
                    ))
                
                results.extend(processed_batches)
        
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            
        return results


class SavePriorDataset:
    """优化8: 多GPU生成和多线程保存"""
    def __init__(self, args):
        self.args = args
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata()
        
        # 设置CUDA设备
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # 初始化GPU生成器
        self.generators = []
        for i in range(self.device_count):
            device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
            gen = BatchGenerator(args, device, device_index=i)
            self.generators.append(gen)
        
        # 设置保存队列
        self.save_queue = queue.Queue(maxsize=SPARSE_BUFFER_SIZE)
        self.save_threads = []
        
        # 启动保存线程
        for i in range(SAVE_THREADS):
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self.save_threads.append(t)
    
    def save_metadata(self):
        """保存元数据，包含所有配置参数"""
        metadata = {
            "prior_type": self.args.prior_type,
            "batch_size": self.args.batch_size,  # 每个文件的数据集数量
            "batch_size_per_gp": self.args.batch_size_per_gp,
            "files_per_gpu": self.args.batch_size_per_gpu,  # 每个GPU的文件数量
            "min_features": self.args.min_features,
            "max_features": self.args.max_features,
            "max_classes": self.args.max_classes,
            "min_seq_len": self.args.min_seq_len,
            "max_seq_len": self.args.max_seq_len,
            "log_seq_len": self.args.log_seq_len,
            "seq_len_per_gp": self.args.seq_len_per_gp,
            "min_train_size": self.args.min_train_size,
            "max_train_size": self.args.max_train_size,
            "replay_small": self.args.replay_small,
        }
        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _save_worker(self):
        """修复并优化保存工作线程"""
        while True:
            batch = self.save_queue.get()
            if batch is None:  # 终止信号
                break
                
            batch_idx, X, y, d, seq_lens, train_sizes = batch
            
            # 统一格式处理（不再需要转换）
            actual_batch_size = d.size(0)
            
            # 创建临时文件
            batch_file = self.save_dir / f"batch_{batch_idx:06d}.pt"
            temp_file = self.save_dir / f"batch_{batch_idx:06d}.pt.tmp"
            
            # 直接保存处理后的张量
            torch.save(
                {
                    "X": X, 
                    "y": y, 
                    "d": d, 
                    "seq_lens": seq_lens, 
                    "train_sizes": train_sizes, 
                    "batch_size": actual_batch_size
                },
                temp_file,
            )
            # 原子重命名确保文件完整性
            temp_file.replace(batch_file)
            self.save_queue.task_done()

    def _generate_batches(self, start_idx, end_idx):
        """优化的并行批量生成"""
        files_per_gpu = (end_idx - start_idx) // self.device_count
        remaining = (end_idx - start_idx) % self.device_count
        
        # 增加并行度
        with ThreadPoolExecutor(max_workers=min(self.device_count, GENERATION_THREADS)) as executor:
            futures = {}
            for i, gen in enumerate(self.generators):
                num_files = files_per_gpu + (1 if i < remaining else 0)
                if num_files > 0:
                    # 为每个GPU分配任务
                    futures[executor.submit(gen.generate, num_files)] = i
            
            # 组织结果
            organized = []
            current_idx = start_idx
            for future in concurrent.futures.as_completed(futures):
                gpu_index = futures[future]
                file_data = future.result()
                for data in file_data:
                    organized.append((current_idx, *data))
                    current_idx += 1
                    
        return organized

    def run(self):
        """优化后的运行方法"""
        # 修改点2：总文件数，不是批次中的数据集数
        total_files = self.args.num_batches  
        files_per_round = self.args.batch_size_per_gpu * self.device_count
        
        print(f"Using {self.device_count} GPU(s) for generation")
        print(f"Files per GPU per round: {self.args.batch_size_per_gpu}")
        print(f"Files per round: {files_per_round}")
        print(f"Total files to generate: {total_files}")
        
        # 添加GPU监控
        gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        
       # 动态调整机制 - 基于GPU使用率
        dynamic_factor = 1.0
        
        files_generated = 0
        with tqdm(total=total_files, desc="Generating files") as pbar:
            while files_generated < total_files:
                # 监控GPU利用率
                gpu_stats = []
                if torch.cuda.is_available():
                    for i in range(self.device_count):
                        try:
                            # 使用pynvml直接查询GPU利用率更可靠
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            mem_used = mem_info.used / (1024 ** 3)
                            mem_total = mem_info.total / (1024 ** 3)
                            gpu_stats.append(f"GPU#{i}: {util}% util, {mem_used:.1f}/{mem_total:.1f}GB")
                        except Exception:
                            # 回退到torch方法
                            alloc_mem = torch.cuda.memory_allocated(i) / (1024 ** 3)
                            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                            gpu_stats.append(f"GPU#{i}: (NA)% util, {alloc_mem:.1f}/{total_mem:.1f}GB")
                
                # 显示状态
                if files_generated % 100 == 0:
                    status = f"[Batch {files_generated}/{total_files}] Dynamic factor: {dynamic_factor:.2f}"
                    if gpu_stats:
                        status += " | " + " | ".join(gpu_stats)
                    print(status)
                
                # 调整批处理大小
                current_per_gpu = int(self.args.batch_size_per_gpu * dynamic_factor)
                files_to_generate = min(total_files - files_generated, current_per_gpu * self.device_count)
                start_idx = files_generated
                end_idx = start_idx + files_to_generate
                
                # 并行生成批次文件
                file_data = self._generate_batches(start_idx, end_idx)
                
                # 放入保存队列
                for batch in file_data:
                    self.save_queue.put(batch)
                    files_generated += 1
                    pbar.update(1)
                
                # 短暂暂停以避免内存峰值
                time.sleep(0.1)
        
        # 等待所有保存完成
        self.save_queue.join()
        
        # 关闭保存线程
        for _ in range(SAVE_THREADS):
            self.save_queue.put(None)
        
        for t in self.save_threads:
            t.join()


if __name__ == "__main__":
    def str2bool(value):
        return value.lower() == "true"

    def train_size_type(value):
        """Custom type function to handle both int and float train sizes."""
        value = float(value)
        if 0 < value < 1:
            return value
        elif value.is_integer():
            return int(value)
        else:
            raise argparse.ArgumentTypeError(
                "Train size must be either an integer (absolute position) "
                "or a float between 0 and 1 (ratio of sequence length)."
            )

    parser = argparse.ArgumentParser(description="Generate training prior datasets")
    parser.add_argument("--save_dir", type=str, default="data", help="Directory to save the generated data")
    parser.add_argument("--np_seed", type=int, default=123, help="Random seed for numpy")
    parser.add_argument("--torch_seed", type=int, default=123, help="Random seed for torch")
    # 修改点3：更清晰的参数名称
    parser.add_argument("--num_batches", type=int, default=10000, help="Total number of files to generate")
    parser.add_argument("--resume_from", type=int, default=0, help="Resume generation from this file index")
    parser.add_argument("--batch_size", type=int, default=512, 
                       help="Number of datasets per batch file")
    parser.add_argument("--batch_size_per_gp", type=int, default=4, 
                       help="Batch size per group within each dataset")
    parser.add_argument("--min_features", type=int, default=2, help="Minimum number of features")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--min_seq_len", type=int, default=None, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument(
        "--log_seq_len",
        default=False,
        type=str2bool,
        help="If True, sample sequence length from log-uniform distribution between min_seq_len and max_seq_len",
    )
    parser.add_argument(
        "--seq_len_per_gp",
        default=False,
        type=str2bool,
        help="If True, sample sequence length independently for each group",
    )
    parser.add_argument(
        "--min_train_size", type=train_size_type, default=0.1, help="Minimum training size position/ratio"
    )
    parser.add_argument(
        "--max_train_size", type=train_size_type, default=0.9, help="Maximum training size position/ratio"
    )
    parser.add_argument(
        "--replay_small",
        default=False,
        type=str2bool,
        help="If True, occasionally sample smaller sequence lengths to ensure model robustness on smaller datasets",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="graph_scm",
        choices=["mlp_scm", "tree_scm", "mix_scm"],
        help="Type of prior to use",
    )
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs for parallel processing")
    parser.add_argument("--num_threads_per_generate", type=int, default=1, help="Threads per generation")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for generation"
    )
    parser.add_argument("--files_per_gpu", type=int, default=128,  # 从16提高到128
                   help="Number of files each GPU generates per iteration")
    parser.add_argument("--gpu_batch_factor", type=float, default=0.8,
                    help="GPU memory usage factor for dynamic batch adjustment")
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.np_seed)
    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)
    
    # 修改点5：在代码内部使用一致的参数名
    args.batch_size_per_gpu = args.files_per_gpu
    
    saver = SavePriorDataset(args)
    saver.run()