# Run the training job

import sagemaker as sage
import boto3

sess = sage.Session()

role = "<your_role_with_sagemaker_enabled>"


# S3 data locations
prefix_path = '/some/prefix/'
bucket_source = sess.default_bucket() + prefix_path
bucket_output = sess.default_bucket() + prefix_path

training_source_file = 'bc_reference.csv'
inference_source_file = 'bc_analysis.csv'

# NannyML parameters
algorith_image_name = "nannyml-algo-poc"
image_uri = '684294718906.dkr.ecr.eu-central-1.amazonaws.com/nannyml-algo-poc:latest'

hyperparameters = {
    "y_pred_proba": "y_pred_proba",
    "y_pred": "y_pred",
    "y_true": "repaid",
    "timestamp_column_name": "timestamp",
    "metrics": ["roc_auc"],
    "chunk_size": 5000,
    "problem_type": "classification_binary",
    "data_filename": training_source_file,
    "data_type": "csv",
}

estimator = sage.estimator.Estimator(
    image,
    role,
    1,
    "ml.m5.large",
    output_path="s3://{}/output".format(bucket_output),
    hyperparameters=hyperparameters,
)

estimator.fit(f'{bucket_source}/{training_source_file}')


transformer = estimator.transformer(instance_count=1, instance_type="ml.m5.large")
transformer.transform(
   f'{bucket_source}/{training_source_file}'
    content_type="text/csv",
)
transformer.wait()
