# Airflow ML Pipeline Deployment Checklist

## 🚀 Pre-Deployment Checklist

### ✅ Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] AWS CLI configured with appropriate credentials
- [ ] AWS S3 bucket created for data storage
- [ ] AWS IAM permissions configured for S3 access
- [ ] Python dependencies verified (torch, transformers, etc.)

### ✅ Local Development Setup
- [ ] Clone/organize project files in correct structure
- [ ] Run `chmod +x setup_airflow.sh` to make setup script executable
- [ ] Run `./setup_airflow.sh` to initialize Airflow
- [ ] Update `.env` file with real AWS credentials
- [ ] Start Airflow services: `docker-compose -f docker-compose.airflow.yml up -d`
- [ ] Access Airflow UI at `http://localhost:8080`
- [ ] Enable DAG: `ml_engagement_prediction_pipeline`
- [ ] Test manual DAG trigger and verify all tasks complete

### ✅ Code Validation
- [ ] Verify `modelfactory/preprocess/preprocess.py` can access S3 data
- [ ] Verify `modelfactory/train.py` can save models to S3
- [ ] Verify `modelfactory/test.py` can load models from S3
- [ ] Test MLflow tracking functionality
- [ ] Confirm all Python imports work in Airflow container

## 🏭 Production Deployment (AWS MWAA)

### ✅ AWS Infrastructure Setup
- [ ] Create S3 bucket for MWAA artifacts: `s3://your-mwaa-bucket`
- [ ] Create VPC and subnets (or use existing)
- [ ] Create security groups for MWAA
- [ ] Create IAM execution role for MWAA with required permissions:
  - [ ] S3 access to data buckets
  - [ ] CloudWatch Logs access
  - [ ] MWAA service permissions

### ✅ MWAA Environment Creation
- [ ] Upload DAG file to S3: `s3://your-mwaa-bucket/dags/`
- [ ] Upload requirements.txt to S3: `s3://your-mwaa-bucket/requirements.txt`
- [ ] Create MWAA environment via AWS Console or CLI
- [ ] Configure environment variables in MWAA:
  - [ ] `S3_BUCKET_NAME`
  - [ ] `AWS_REGION`
- [ ] Wait for environment creation (20-30 minutes)

### ✅ Testing and Validation
- [ ] Access MWAA Airflow UI through AWS Console
- [ ] Verify DAG appears and is syntactically correct
- [ ] Enable DAG in MWAA UI
- [ ] Trigger manual DAG run
- [ ] Monitor task execution and logs
- [ ] Verify data flows correctly through S3
- [ ] Confirm model training and saving works
- [ ] Check MLflow tracking integration

## 📊 Monitoring and Alerting Setup

### ✅ CloudWatch Configuration
- [ ] Enable CloudWatch logging for all MWAA components
- [ ] Create CloudWatch dashboard for MWAA metrics
- [ ] Set up CloudWatch alarms for task failures
- [ ] Set up CloudWatch alarms for DAG execution time

### ✅ Notification Setup
- [ ] Configure SMTP settings for email notifications
- [ ] Test email notifications for success/failure
- [ ] (Optional) Set up Slack webhook integration
- [ ] Verify alert email addresses are correct

### ✅ MLflow Setup
- [ ] Ensure MLflow tracking server is accessible
- [ ] Verify experiment logging works from MWAA
- [ ] Check model artifact storage in S3
- [ ] Test model versioning and comparison features

## 🔄 Weekly Schedule Verification

### ✅ Schedule Configuration
- [ ] Verify cron expression: `0 2 * * 1` (Every Monday 2AM UTC)
- [ ] Confirm timezone settings (UTC)
- [ ] Test schedule by temporarily setting it to run soon
- [ ] Verify `catchup=False` to prevent running missed schedules
- [ ] Confirm `max_active_runs=1` to prevent overlapping executions

## 🛡️ Security and Best Practices

### ✅ Security Checklist
- [ ] Use IAM roles instead of hardcoded credentials
- [ ] Implement least-privilege access policies
- [ ] Encrypt S3 buckets and enable versioning
- [ ] Use VPC endpoints for S3 access if possible
- [ ] Enable AWS CloudTrail for audit logging

### ✅ Code Quality
- [ ] Implement proper error handling in all tasks
- [ ] Add data validation checks before training
- [ ] Implement idempotent task operations
- [ ] Add comprehensive logging throughout pipeline
- [ ] Use resource pools to limit concurrent GPU usage

## 📈 Performance Optimization

### ✅ Resource Management
- [ ] Choose appropriate MWAA environment class (mw1.small/medium/large)
- [ ] Configure task timeouts appropriately
- [ ] Set up resource pools for GPU-intensive tasks
- [ ] Monitor memory usage and adjust instance sizes
- [ ] Implement data chunking for large datasets

### ✅ Cost Optimization
- [ ] Use spot instances where appropriate
- [ ] Implement S3 lifecycle policies for old data
- [ ] Monitor MWAA costs and usage patterns
- [ ] Set up cost alerts for unexpected spikes

## 🔍 Post-Deployment Monitoring

### ✅ First Week Monitoring
- [ ] Monitor first scheduled run completion
- [ ] Check CloudWatch logs for any warnings
- [ ] Verify model training metrics are reasonable
- [ ] Confirm S3 data is being updated correctly
- [ ] Check MLflow experiments are being logged

### ✅ Ongoing Maintenance
- [ ] Weekly review of DAG execution logs
- [ ] Monthly review of model performance metrics
- [ ] Quarterly review of infrastructure costs
- [ ] Regular updates to Python dependencies
- [ ] Backup of important S3 data and models

## 🚨 Troubleshooting Quick Reference

### Common Issues
- **DAG Import Errors**: Check Python path and dependencies
- **Task Timeouts**: Increase timeout values or optimize code
- **S3 Access Denied**: Verify IAM permissions and bucket policies
- **Out of Memory**: Reduce batch sizes or increase instance size
- **GPU Not Available**: Check instance type and CUDA availability

### Emergency Contacts
- [ ] Document who to contact for AWS issues
- [ ] Document who to contact for ML model issues
- [ ] Document escalation procedures
- [ ] Keep contact information updated

## ✅ Final Go-Live Checklist

- [ ] All automated tests pass
- [ ] Manual end-to-end test successful
- [ ] Monitoring and alerting configured
- [ ] Team trained on Airflow UI and troubleshooting
- [ ] Documentation updated and accessible
- [ ] Rollback plan documented and tested
- [ ] Stakeholders notified of go-live schedule

---

**Note**: This checklist should be customized based on your specific requirements and environment. Keep it updated as your pipeline evolves. 