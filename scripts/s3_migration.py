#!/usr/bin/env python3
"""
S3 Migration Script for Social Media Analyzer Data

This script migrates TikTok data and thumbnails from local storage to S3
with the following structure:
s3://socialmediaanalyzer/
├── raw/
│   ├── data/tiktok_data.parquet
│   ├── thumbnails/video123.jpg
"""

import os
import boto3
import logging
from pathlib import Path
from tqdm import tqdm
from botocore.exceptions import ClientError, NoCredentialsError
import argparse


class S3DataMigrator:
    def __init__(self, bucket_name, aws_profile=None):
        """
        Initialize S3 client and migration settings
        
        Args:
            bucket_name (str): Name of the S3 bucket
            aws_profile (str, optional): AWS profile to use
        """
        self.bucket_name = bucket_name
        self.setup_logging()
        self.setup_s3_client(aws_profile)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('s3_migration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_s3_client(self, aws_profile):
        """Setup S3 client with optional profile"""
        try:
            if aws_profile:
                session = boto3.Session(profile_name=aws_profile)
                self.s3_client = session.client('s3')
            else:
                self.s3_client = boto3.client('s3')
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure your credentials.")
            raise
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                self.logger.error(f"Bucket {self.bucket_name} not found.")
                raise
            else:
                self.logger.error(f"Error connecting to S3: {e}")
                raise
    
    def upload_file_to_s3(self, local_file_path, s3_key):
        """
        Upload a file to S3
        
        Args:
            local_file_path (str): Path to local file
            s3_key (str): S3 object key (path)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get file size for progress tracking
            file_size = os.path.getsize(local_file_path)
            
            # Upload with progress callback
            self.s3_client.upload_file(
                local_file_path, 
                self.bucket_name, 
                s3_key,
                Callback=lambda bytes_transferred: None  # Could add progress here
            )
            
            self.logger.info(f"Uploaded {local_file_path} -> s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_file_path}: {e}")
            return False
    
    def migrate_parquet_data(self, data_folder_path):
        """
        Migrate parquet data files to S3
        
        Args:
            data_folder_path (str): Path to local data folder
        
        Returns:
            bool: True if successful, False otherwise
        """
        parquet_file = os.path.join(data_folder_path, 'tiktok_data', 'tiktok_data.parquet')
        
        if not os.path.exists(parquet_file):
            self.logger.error(f"Parquet file not found: {parquet_file}")
            return False
        
        s3_key = 'raw/data/tiktok_data.parquet'
        self.logger.info(f"Migrating parquet data: {parquet_file}")
        
        return self.upload_file_to_s3(parquet_file, s3_key)
    
    def migrate_thumbnails(self, data_folder_path):
        """
        Migrate thumbnail images to S3
        
        Args:
            data_folder_path (str): Path to local data folder
        
        Returns:
            tuple: (success_count, total_count)
        """
        thumbnails_folder = os.path.join(data_folder_path, 'tiktok_data', 'thumbnails')
        
        if not os.path.exists(thumbnails_folder):
            self.logger.error(f"Thumbnails folder not found: {thumbnails_folder}")
            return 0, 0
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(thumbnails_folder).glob(ext))
        
        if not image_files:
            self.logger.warning("No image files found in thumbnails folder")
            return 0, 0
        
        self.logger.info(f"Found {len(image_files)} thumbnail images to migrate")
        
        success_count = 0
        total_count = len(image_files)
        
        # Upload with progress bar
        with tqdm(total=total_count, desc="Uploading thumbnails") as pbar:
            for image_file in image_files:
                s3_key = f'raw/thumbnails/{image_file.name}'
                
                if self.upload_file_to_s3(str(image_file), s3_key):
                    success_count += 1
                
                pbar.update(1)
        
        return success_count, total_count
    
    def verify_migration(self):
        """
        Verify that files were uploaded successfully
        
        Returns:
            dict: Summary of verification results
        """
        verification_results = {
            'parquet_data': False,
            'thumbnails_count': 0,
            'total_objects': 0
        }
        
        try:
            # List objects in the bucket with our prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix='raw/'
            )
            
            if 'Contents' not in response:
                self.logger.warning("No objects found in S3 bucket")
                return verification_results
            
            for obj in response['Contents']:
                key = obj['Key']
                verification_results['total_objects'] += 1
                
                if key == 'raw/data/tiktok_data.parquet':
                    verification_results['parquet_data'] = True
                elif key.startswith('raw/thumbnails/') and key.endswith('.jpg'):
                    verification_results['thumbnails_count'] += 1
            
            self.logger.info(f"Verification results: {verification_results}")
            return verification_results
        
        except Exception as e:
            self.logger.error(f"Error during verification: {e}")
            return verification_results
    
    def run_migration(self, data_folder_path):
        """
        Run the complete migration process
        
        Args:
            data_folder_path (str): Path to local data folder
        
        Returns:
            dict: Migration summary
        """
        self.logger.info("Starting S3 migration process...")
        
        # Migrate parquet data
        parquet_success = self.migrate_parquet_data(data_folder_path)
        
        # Migrate thumbnails
        thumbnails_success, thumbnails_total = self.migrate_thumbnails(data_folder_path)
        
        # Verify migration
        verification = self.verify_migration()
        
        # Summary
        summary = {
            'parquet_migrated': parquet_success,
            'thumbnails_migrated': thumbnails_success,
            'thumbnails_total': thumbnails_total,
            'verification': verification
        }
        
        self.logger.info("Migration Summary:")
        self.logger.info(f"  Parquet data: {'✓' if parquet_success else '✗'}")
        self.logger.info(f"  Thumbnails: {thumbnails_success}/{thumbnails_total}")
        self.logger.info(f"  Verified objects in S3: {verification['total_objects']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Migrate TikTok data to S3')
    parser.add_argument('--bucket', default='socialmediaanalyzer', required=True, help='S3 bucket name')
    parser.add_argument('--data-folder', default='./modelfactory/data', 
                        help='Path to local data folder')
    parser.add_argument('--aws-profile', help='AWS profile to use')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Show what would be uploaded without actually uploading')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be uploaded")
        print(f"Would migrate data from: {args.data_folder}")
        print(f"Would upload to bucket: {args.bucket}")
        return
    
    try:
        migrator = S3DataMigrator(args.bucket, args.aws_profile)
        summary = migrator.run_migration(args.data_folder)
        
        if summary['parquet_migrated'] and summary['thumbnails_migrated'] > 0:
            print("\n✓ Migration completed successfully!")
        else:
            print("\n⚠ Migration completed with some issues. Check the logs.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
        exit(1)


if __name__ == "__main__":
    main() 