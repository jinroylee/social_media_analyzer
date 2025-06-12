# #!/usr/bin/env python3
# """
# MLflow Experiment Management Script

# This script provides utilities to manage MLflow experiments, start the UI server,
# and organize experiment results.
# """

# import argparse
# import os
# import subprocess
# import sys
# from pathlib import Path
# import mlflow
# from mlflow.tracking import MlflowClient

# def setup_mlflow():
#     """Setup MLflow tracking directory and configuration."""
#     mlflow_dir = Path.cwd() / "mlruns"
#     mlflow_dir.mkdir(exist_ok=True)
    
#     # Set tracking URI to local directory
#     tracking_uri = f"file://{mlflow_dir.absolute()}"
#     mlflow.set_tracking_uri(tracking_uri)
    
#     print(f"MLflow tracking URI set to: {tracking_uri}")
#     return tracking_uri

# def start_ui(host="127.0.0.1", port=5000):
#     """Start MLflow UI server."""
#     tracking_uri = setup_mlflow()
    
#     print(f"Starting MLflow UI server...")
#     print(f"Access the UI at: http://{host}:{port}")
#     print("Press Ctrl+C to stop the server")
    
#     try:
#         subprocess.run([
#             sys.executable, "-m", "mlflow", "ui",
#             "--backend-store-uri", tracking_uri,
#             "--host", host,
#             "--port", str(port)
#         ])
#     except KeyboardInterrupt:
#         print("\nMLflow UI server stopped.")

# def list_experiments():
#     """List all MLflow experiments."""
#     setup_mlflow()
#     client = MlflowClient()
    
#     experiments = client.search_experiments()
    
#     if not experiments:
#         print("No experiments found.")
#         return
    
#     print("Available Experiments:")
#     print("-" * 50)
#     for exp in experiments:
#         print(f"ID: {exp.experiment_id}")
#         print(f"Name: {exp.name}")
#         print(f"Lifecycle Stage: {exp.lifecycle_stage}")
#         if exp.tags:
#             print(f"Tags: {exp.tags}")
#         print("-" * 50)

# def list_runs(experiment_name="social_media_engagement_prediction", max_results=10):
#     """List recent runs from an experiment."""
#     setup_mlflow()
#     client = MlflowClient()
    
#     try:
#         experiment = client.get_experiment_by_name(experiment_name)
#         if not experiment:
#             print(f"Experiment '{experiment_name}' not found.")
#             return
        
#         runs = client.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             max_results=max_results,
#             order_by=["start_time DESC"]
#         )
        
#         if not runs:
#             print(f"No runs found in experiment '{experiment_name}'.")
#             return
        
#         print(f"Recent runs from experiment '{experiment_name}':")
#         print("-" * 80)
        
#         for run in runs:
#             print(f"Run ID: {run.info.run_id}")
#             print(f"Run Name: {run.data.tags.get('mlflow.runName', 'N/A')}")
#             print(f"Status: {run.info.status}")
#             print(f"Start Time: {run.info.start_time}")
            
#             # Print key metrics
#             if run.data.metrics:
#                 print("Key Metrics:")
#                 for metric, value in run.data.metrics.items():
#                     if metric in ['test_mae', 'best_val_mae', 'val_mae', 'test_r2', 'best_val_r2']:
#                         print(f"  {metric}: {value:.4f}")
            
#             # Print tags
#             if run.data.tags:
#                 print("Tags:")
#                 for tag, value in run.data.tags.items():
#                     if not tag.startswith('mlflow.'):
#                         print(f"  {tag}: {value}")
            
#             print("-" * 80)
            
#     except Exception as e:
#         print(f"Error listing runs: {e}")

# def compare_runs(run_ids):
#     """Compare multiple runs by their metrics."""
#     setup_mlflow()
#     client = MlflowClient()
    
#     runs_data = []
#     for run_id in run_ids:
#         try:
#             run = client.get_run(run_id)
#             runs_data.append(run)
#         except Exception as e:
#             print(f"Error fetching run {run_id}: {e}")
#             continue
    
#     if not runs_data:
#         print("No valid runs found.")
#         return
    
#     print("Run Comparison:")
#     print("=" * 100)
    
#     # Print header
#     print(f"{'Run ID':<36} {'Run Name':<25} {'MAE':<8} {'R²':<8} {'Correlation':<12}")
#     print("-" * 100)
    
#     for run in runs_data:
#         run_id = run.info.run_id
#         run_name = run.data.tags.get('mlflow.runName', 'N/A')[:24]
        
#         # Get metrics (prefer test metrics, fall back to validation)
#         mae = (run.data.metrics.get('test_mae') or 
#                run.data.metrics.get('best_val_mae') or 
#                run.data.metrics.get('val_mae', 0))
#         r2 = (run.data.metrics.get('test_r2') or 
#               run.data.metrics.get('best_val_r2') or 
#               run.data.metrics.get('val_r2', 0))
#         corr = (run.data.metrics.get('test_correlation') or 
#                 run.data.metrics.get('val_correlation', 0))
        
#         print(f"{run_id:<36} {run_name:<25} {mae:<8.4f} {r2:<8.4f} {corr:<12.4f}")

# def clean_experiments():
#     """Clean up old and failed experiments."""
#     setup_mlflow()
#     client = MlflowClient()
    
#     print("Cleaning up failed and old experiments...")
    
#     experiments = client.search_experiments()
#     for exp in experiments:
#         runs = client.search_runs(
#             experiment_ids=[exp.experiment_id],
#             filter_string="attributes.status = 'FAILED'"
#         )
        
#         if runs:
#             print(f"Found {len(runs)} failed runs in experiment '{exp.name}'")
#             for run in runs:
#                 print(f"  Deleting failed run: {run.info.run_id}")
#                 client.delete_run(run.info.run_id)

# def export_best_model(experiment_name="social_media_engagement_prediction", 
#                      output_dir="exported_models"):
#     """Export the best model from an experiment."""
#     setup_mlflow()
#     client = MlflowClient()
    
#     try:
#         experiment = client.get_experiment_by_name(experiment_name)
#         if not experiment:
#             print(f"Experiment '{experiment_name}' not found.")
#             return
        
#         # Find the run with the best validation MAE
#         runs = client.search_runs(
#             experiment_ids=[experiment.experiment_id],
#             filter_string="metrics.best_val_mae > 0",
#             order_by=["metrics.best_val_mae ASC"],
#             max_results=1
#         )
        
#         if not runs:
#             print("No runs with validation metrics found.")
#             return
        
#         best_run = runs[0]
#         run_id = best_run.info.run_id
#         best_mae = best_run.data.metrics.get('best_val_mae', 'N/A')
        
#         print(f"Best run found:")
#         print(f"  Run ID: {run_id}")
#         print(f"  Validation MAE: {best_mae}")
        
#         # Create output directory
#         output_path = Path(output_dir)
#         output_path.mkdir(exist_ok=True)
        
#         # Download artifacts
#         artifacts_path = client.download_artifacts(run_id, "", str(output_path / run_id))
#         print(f"Model artifacts downloaded to: {artifacts_path}")
        
#     except Exception as e:
#         print(f"Error exporting model: {e}")

# def main():
#     parser = argparse.ArgumentParser(description="MLflow Experiment Management")
#     subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
#     # UI command
#     ui_parser = subparsers.add_parser('ui', help='Start MLflow UI server')
#     ui_parser.add_argument('--host', default='127.0.0.1', help='Host address')
#     ui_parser.add_argument('--port', type=int, default=5000, help='Port number')
    
#     # List experiments command
#     subparsers.add_parser('list-experiments', help='List all experiments')
    
#     # List runs command
#     runs_parser = subparsers.add_parser('list-runs', help='List recent runs')
#     runs_parser.add_argument('--experiment', default='social_media_engagement_prediction',
#                            help='Experiment name')
#     runs_parser.add_argument('--max-results', type=int, default=10,
#                            help='Maximum number of runs to show')
    
#     # Compare runs command
#     compare_parser = subparsers.add_parser('compare', help='Compare multiple runs')
#     compare_parser.add_argument('run_ids', nargs='+', help='Run IDs to compare')
    
#     # Clean command
#     subparsers.add_parser('clean', help='Clean up failed experiments')
    
#     # Export command
#     export_parser = subparsers.add_parser('export-best', help='Export best model')
#     export_parser.add_argument('--experiment', default='social_media_engagement_prediction',
#                               help='Experiment name')
#     export_parser.add_argument('--output-dir', default='exported_models',
#                               help='Output directory')
    
#     args = parser.parse_args()
    
#     if args.command == 'ui':
#         start_ui(args.host, args.port)
#     elif args.command == 'list-experiments':
#         list_experiments()
#     elif args.command == 'list-runs':
#         list_runs(args.experiment, args.max_results)
#     elif args.command == 'compare':
#         compare_runs(args.run_ids)
#     elif args.command == 'clean':
#         clean_experiments()
#     elif args.command == 'export-best':
#         export_best_model(args.experiment, args.output_dir)
#     else:
#         parser.print_help()

# if __name__ == "__main__":
#     main() 