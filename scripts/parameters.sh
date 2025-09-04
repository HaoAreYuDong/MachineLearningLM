#!/bin/bash

# =============================================================================
# Unified Parameters Configuration for Data Processing Pipeline
# =============================================================================
# This file supports both single-threaded and multi-processing workflows
# =============================================================================

# Required parameters


export dataset_names="abalone accelerometer ada ada_agnostic ada_prior adult airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True allbp allrep Amazon_employee_access analcatdata_authorship artificial_characters ASP_POTASSCO_classification autoUniv_au4_2500 autoUniv_au7_1100 banknote_authentication Bank_Customer_Churn_Dataset baseball Basketball_c BNG_breast_w BNG_cmc BNG_tic_tac_toe California_Housing_Classification car_evaluation Cardiovascular_Disease_dataset Click_prediction_small company_bankruptcy_prediction compass connect_4 Contaminant_detection_in_packaged_cocoa_hazelnut_spread_jars_using_Microwaves_Sensing_and_Machine_Learning_10_5GHz_Urbinati Contaminant_detection_in_packaged_cocoa_hazelnut_spread_jars_using_Microwaves_Sensing_and_Machine_Learning_11_0GHz_Urbinati Contaminant_detection_in_packaged_cocoa_hazelnut_spread_jars_using_Microwaves_Sensing_and_Machine_Learning_9_0GHz_Urbinati contraceptive_method_choice credit Credit_c Customer_Personality_Analysis customer_satisfaction_in_airline dabetes_130_us_hospitals default_of_credit_card_clients delta_ailerons Diabetic_Retinopathy_Debrecen dis dna drug_consumption dry_bean_dataset E_CommereShippingData eeg_eye_state electricity Employee estimation_of_obesity_levels eucalyptus eye_movements eye_movements_bin Firm_Teacher_Clave_Direction_Classification first_order_theorem_proving Fitness_Club_c FOREX_audcad_day_High FOREX_audcad_hour_High FOREX_audjpy_hour_High FOREX_audsgd_hour_High FOREX_audusd_hour_High FOREX_cadjpy_day_High FOREX_cadjpy_hour_High GAMETES_Epistasis_2_Way_20atts_0_1H_EDM_1_1 gas_drift Gender_Gap_in_Spanish_WP GesturePhaseSegmentationProcessed golf_play_dataset_extended Heart_Disease_Dataset_Comprehensive helena hill_valley house_16H HR_Analytics_Job_Change_of_Data_Scientists htru ibm_employee_performance Indian_pines INNHotelsGroup Insurance internet_firewall jasmine internet_usage Intersectional_Bias_Assessment in_vehicle_coupon_recommendation Is_this_a_good_customer JapaneseVowels jm1 jungle_chess_2pcs_raw_endgame_complete KDD KDDCup09_upselling kdd_ipums_la_97_small kr_vs_k kr_vs_kp kropt law_school_admission_bianry letter Long madeline MagicTelescope mammography Marketing_Campaign mfeat_factors mfeat_fourier mfeat_karhunen mfeat_morphological mfeat_pixel mfeat_zernike MIC mice_protein_expression microaggregation2 mobile_c36_oversampling Mobile_Price_Classification mozilla4 naticusdroid_android_permissions_dataset national_longitudinal_survey_binary National_Health_and_Nutrition_Health_Survey okcupid_stem one_hundred_plants_margin one_hundred_plants_shape one_hundred_plants_texture online_shoppers optdigits ozone_level_8hr ozone_level page_blocks pc3 pc4 pendigits Performance_Prediction philippine PhishingWebsites PieChart3 PizzaCutter3 pol predict_students_dropout_and_academic_success Pumpkin_Seeds QSAR_biodegradation Rain_in_Australia rice_cammeo_and_osmancik satimage SDSS17 semeion shill_bidding Shipping shrutime shuttle spambase splice sports_articles_for_objectivity_analysis steel_plates_faults svmguide3 sylvine taiwanese_bankruptcy_prediction telco_customer_churn Telecom_Churn_Dataset texture thyroid_ann thyroid_dis turiye_student_evaluation UJI_Pen_Characters VulNoneVul walking_activity Waterstress water_quality Water_Quality_and_Potability waveform_5000 website_phishing Wilt wine_quality_red wine_quality_white blood"

export train_chunk_sizes="8 16 32 64 128 256 512 1024"
export row_shuffle_seeds="40"

# Directory parameters (with clear naming)
export original_data_dir="../llm4mle/datahub_inputs/data_raw"     # Input for data_chunk_prep.py
export datahub_outputs_dir="/storage/v-mingzhelu/Res_mle_configs/all_170_knn"             # Base directory for all outputs

export split_data_dir="$datahub_outputs_dir/1_split"       
export prompt_data_dir="$datahub_outputs_dir/2_prompt"     
export predict_data_dir="$datahub_outputs_dir/3_predict"   
export metric_data_dir="$datahub_outputs_dir/4_metric"     
export report_data_dir="$datahub_outputs_dir/5_report"     

# Common parameters
export split_seed=42
export test_chunk_size=50
export max_workers=8      # Used by single scripts
export force_overwrite=True

# Multi-processing specific parameters (only used by mp/ scripts)
export max_parallel_jobs=20  # Maximum number of (chunk_size, seed) combinations to run in parallel
export wait_between_datasets=1  # Seconds to wait between datasets (for system stability)

# Data preparation specific parameters
export test_size=0.2
export shuffle_columns=True

# Prompt generation specific parameters
export normalization=True
export include_feature_descriptions=False
export prompt_format_style="concat"

# Model prediction specific parameters
# export model_name="minzl/toy3_2800"
# export model_name="openai::gpt-4o-mini"   # API example
# export model_name="randomforest"     # ML example
# export model_name="xgboost"          # ML example  
# export model_name="knn"              # ML example


export temperature=0.0
export max_samples=""  # Empty means no limit
export labels=""       # Empty means auto-detect from data
export device_id="0"   # GPU device ID to use (default: "0")
export logprobs_supported="True"  # Whether the model supports logprobs

# =============================================================================
# Parameter Descriptions:
# =============================================================================
# dataset_names:              Space-separated list of dataset names to process
# train_chunk_sizes:          Space-separated list of training chunk sizes to use
# row_shuffle_seeds:          Space-separated list of row shuffle seeds to use
# original_data_dir:          Directory containing raw CSV data files
# datahub_outputs_dir:        Base directory for all pipeline outputs
# split_data_dir:             Directory for split data (output of prep, input of prompt_gen)
# prompt_data_dir:            Directory for generated prompts (output of prompt_gen, input of model_pred)
# predict_data_dir:           Directory for model predictions (output of model_pred, input of evaluator)
# metric_data_dir:            Directory for evaluation metrics (output of evaluator)
# report_data_dir:            Directory for summary reports (output of metric_summarizer)
# split_seed:                 Random seed for train/test split
# test_chunk_size:            Size of test data chunks
# max_workers:                Number of parallel workers for processing (single scripts)
# force_overwrite:            Force overwrite existing directories without prompting (True/False)
# max_parallel_jobs:          Maximum number of (chunk_size, seed) combinations to run simultaneously (mp scripts)
# wait_between_datasets:      Seconds to wait between processing different datasets (mp scripts)
# test_size:                  Proportion of data to use for testing (0.0-1.0) [data_prep only]
# shuffle_columns:            Whether to shuffle column order (True/False) [data_prep only]
# normalization:              Apply normalization to the data (True/False) [prompt_gen only]
# include_feature_descriptions: Include feature descriptions in prompts (True/False) [prompt_gen only]
# prompt_format_style:        Data formatting style: 'concat' or 'tabllm' [prompt_gen only]
# model_name:                 Model to use for prediction:
#                               - ML models: randomforest, xgboost, knn (auto-detected)
#                               - DL models: minzl/toy3_2800, etc. (auto-detected)  
#                               - API models: openai::gpt-4o, etc. (auto-detected)
#                             [model_pred only]
# temperature:                Sampling temperature for model [model_pred only]
# max_samples:                Maximum number of samples to process (empty = no limit) [model_pred only]
# labels:                     Comma-separated labels (empty = auto-detect) [model_pred only]
# device_id:                  GPU device ID to use (e.g., "0", "1", "2") [model_pred only]
# =============================================================================

echo "SUCCESS: Unified parameters loaded successfully!"
echo ""
echo "📋 Available Scripts:"
echo "   Single-threaded:"
echo "     - Data Prep: './single/data_prep.sh'"
echo "     - Prompt Gen: './single/prompt_gen.sh'"
echo "     - Model Pred: './single/model_pred.sh' (auto-detects ML/DL)"
echo "     - Evaluation: './single/evaluation.sh'"
echo "     - Report: './single/report.sh'"
echo ""
echo "   Multi-processing:"
echo "     - Data Prep: './mp/data_prep_mp.sh'"
echo "     - Prompt Gen: './mp/prompt_gen_mp.sh'"
echo "     - Model Pred: './mp/model_pred.sh' (auto-detects ML/DL/API)"
echo "     - Evaluation: './mp/evaluation_mp.sh'"
echo "     - Report: './mp/report_mp.sh'"
echo ""
echo "   End-to-end Pipeline:"
echo "     - Server Pipeline: './server_pipeline.sh' (auto-detects ML/DL/API)"
echo ""
echo "📊 Current Configuration:"
echo "   - Datasets: $dataset_names"
echo "   - Chunk sizes: $train_chunk_sizes" 
echo "   - Row shuffle seeds: $row_shuffle_seeds"
echo "   - Model: $model_name"
echo "   - GPU device: $device_id"

