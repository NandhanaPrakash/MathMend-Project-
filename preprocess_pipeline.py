"""
preprocess_pipeline.py
-----------------------
Complete preprocessing pipeline for math word problems.

Pipeline Sequence:
1. Data Loading → 2. Text Cleaning → 3. Equation Canonicalization → 
4. Metadata Extraction → 5. Feature Engineering → 6. Save Results

Author: Math Retrieval System
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List
import json

# Add the current directory to path to import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from pipeline_sequence.data_loader import DataLoader
from pipeline_sequence.cleaning import clean_dataframe, cleaning_report
from pipeline_sequence.canonicalizer import canonicalize_dataframe
from pipeline_sequence.metadata_extractor import extract_metadata_dataframe
from pipeline_sequence.features import compute_structure_features_df

# Configuration
class Config:
    """Configuration for the preprocessing pipeline."""
    
    # Input/Output paths
    RAW_DATA_PATH = "data/raw/word_problems.csv"
    OUTPUT_DIR = "output/preprocessed"
    LOGS_DIR = "logs"
    
    # Column names in your dataset
    PROBLEM_ID_COL = "problem_id"
    PROBLEM_TEXT_COL = "problem_text" 
    SOLUTION_TEXT_COL = "solution_text"
    
    # Processing parameters
    CLEANED_COL = "clean_text"
    FINGERPRINT_COL = "symbolic_fingerprint"
    PARSED_COL = "parsed_equations"
    METADATA_COL = "metadata_dict"
    STRUCTURE_VECTOR_COL = "structure_vector"
    
    # Feature engineering
    WL_ITERATIONS = 2
    WL_FEATURE_SIZE = 64
    MAX_VARIABLES = 6
    STRUCTURE_VECTOR_SIZE = 256

# Setup logging
def setup_logging():
    """Configure logging for the pipeline."""
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    log_filename = f"{Config.LOGS_DIR}/preprocess_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

class PreprocessPipeline:
    """
    Complete preprocessing pipeline for math word problems.
    """
    
    def __init__(self):
        self.logger = setup_logging()
        self.df = None
        self.stats = {}
        
    def run(self, input_path: str = None) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            input_path: Path to input CSV file. Uses config path if None.
            
        Returns:
            Preprocessed DataFrame with all features
        """
        self.logger.info("🚀 Starting Preprocessing Pipeline")
        self.logger.info(f"Input: {input_path or Config.RAW_DATA_PATH}")
        
        try:
            # Step 1: Data Loading
            self._load_data(input_path)
            
            # Step 2: Text Cleaning
            self._clean_text()
            
            # Step 3: Equation Canonicalization
            self._canonicalize_equations()
            
            # Step 4: Metadata Extraction
            self._extract_metadata()
            
            # Step 5: Feature Engineering
            self._compute_structure_features()
            
            # Step 6: Save Results
            self._save_results()
            
            self.logger.info("✅ Preprocessing Pipeline Completed Successfully!")
            self._print_summary()
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
    
    def _load_data(self, input_path: str = None):
        """Step 1: Load and validate the dataset."""
        self.logger.info("📥 Step 1: Loading Data...")
        
        data_path = input_path or Config.RAW_DATA_PATH
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file not found: {data_path}")
        
        # Load data using your DataLoader
        loader = DataLoader(
            file_path=data_path,
            required_columns=[Config.PROBLEM_ID_COL, Config.PROBLEM_TEXT_COL]
        )
        
        self.df = loader.load()
        self.stats['original_count'] = len(self.df)
        
        self.logger.info(f"   Loaded {len(self.df)} problems")
        self.logger.info(f"   Columns: {list(self.df.columns)}")
        self.logger.info(f"   Sample problem: {self.df[Config.PROBLEM_TEXT_COL].iloc[0][:100]}...")
    
    def _clean_text(self):
        """Step 2: Clean and normalize problem text."""
        self.logger.info("🧹 Step 2: Cleaning Text...")
        
        original_texts = self.df[Config.PROBLEM_TEXT_COL].copy()
        
        # Use your cleaning module
        self.df = clean_dataframe(self.df, Config.PROBLEM_TEXT_COL)
        
        # Generate cleaning report
        report = cleaning_report(
            self.df, 
            raw_col=Config.PROBLEM_TEXT_COL, 
            clean_col=Config.CLEANED_COL
        )
        
        self.stats['cleaning_report'] = report
        self.stats['cleaned_count'] = len(self.df)
        
        self.logger.info(f"   Cleaned {report['changed_texts']} texts")
        self.logger.info(f"   Empty after cleaning: {report['empty_after_clean']}")
        
        # Show cleaning example
        if len(self.df) > 0:
            self.logger.info(f"   Cleaning example:")
            self.logger.info(f"     Before: {original_texts.iloc[0][:80]}...")
            self.logger.info(f"     After:  {self.df[Config.CLEANED_COL].iloc[0][:80]}...")
    
    def _canonicalize_equations(self):
        """Step 3: Advanced equation extraction and canonicalization."""
        self.logger.info("🔍 Step 3: Advanced Equation Canonicalization...")
        
        try:
            from pipeline_sequence.advanced_equation_extractor import extract_equations_advanced
            from pipeline_sequence.canonicalizer import canonicalize_system
            
            extracted_equations = []
            fingerprints = []
            parsed_data = []
            
            for idx, row in self.df.iterrows():
                problem_text = row[Config.CLEANED_COL]
                
                # Use ADVANCED equation extractor
                equations = extract_equations_advanced(problem_text)
                
                # Canonicalize the extracted equations
                canonicalized = canonicalize_system(equations)
                
                extracted_equations.append(equations)
                fingerprints.append(canonicalized['fingerprint'])
                parsed_data.append(canonicalized)
            
            self.df['extracted_equations'] = extracted_equations
            self.df['symbolic_fingerprint'] = fingerprints
            self.df['parsed_equations'] = parsed_data
            
            # Calculate statistics
            valid_fingerprints = self.df[Config.FINGERPRINT_COL].notna().sum()
            unique_fingerprints = self.df[Config.FINGERPRINT_COL].nunique()
            
            self.stats['canonicalization'] = {
                'valid_fingerprints': int(valid_fingerprints),
                'unique_fingerprints': int(unique_fingerprints),
                'success_rate': float(valid_fingerprints / len(self.df))
            }
            
            self.logger.info(f"   Successfully canonicalized: {valid_fingerprints}/{len(self.df)}")
            self.logger.info(f"   Unique equation fingerprints: {unique_fingerprints}")
            
            # Show canonicalization example
            if valid_fingerprints > 0:
                sample_idx = self.df[Config.FINGERPRINT_COL].notna().idxmax()
                self.logger.info(f"   Canonicalization example:")
                self.logger.info(f"     Problem: {self.df[Config.CLEANED_COL].iloc[sample_idx][:60]}...")
                self.logger.info(f"     Fingerprint: {self.df[Config.FINGERPRINT_COL].iloc[sample_idx]}")
                
        except Exception as e:
            self.logger.error(f"Advanced canonicalization failed: {e}")
            # Fallback to basic
            from pipeline_sequence.equation_extractor import extract_equations_from_problem
            self.df = canonicalize_dataframe(
                self.df,
                source_col=Config.CLEANED_COL,
                reasoning_col=None,
                equations_col_out="extracted_equations",
                fingerprint_col=Config.FINGERPRINT_COL,
                parsed_col=Config.PARSED_COL
            )
            
            # Calculate statistics for fallback
            valid_fingerprints = self.df[Config.FINGERPRINT_COL].notna().sum()
            unique_fingerprints = self.df[Config.FINGERPRINT_COL].nunique()
            
            self.stats['canonicalization'] = {
                'valid_fingerprints': int(valid_fingerprints),
                'unique_fingerprints': int(unique_fingerprints),
                'success_rate': float(valid_fingerprints / len(self.df))
            }
    def _extract_metadata(self):
        """Step 4: Extract metadata and problem characteristics."""
        self.logger.info("📊 Step 4: Extracting Metadata...")
        
        self.df = extract_metadata_dataframe(self.df)
        
        # Statistics
        problem_types = self.df['problem_type'].value_counts().to_dict()
        difficulty_counts = self.df['difficulty_level'].value_counts().to_dict()
        
        self.stats['metadata'] = {
            'problem_types': problem_types,
            'difficulty_distribution': difficulty_counts,
            'avg_equations': float(self.df['equation_count'].mean()),
            'avg_variables': float(self.df['variable_count'].mean())
        }
        
        self.logger.info(f"   Problem types: {problem_types}")
        self.logger.info(f"   Difficulty: {difficulty_counts}")
        self.logger.info(f"   Avg equations: {self.df['equation_count'].mean():.2f}")
        self.logger.info(f"   Avg variables: {self.df['variable_count'].mean():.2f}")
    
    def _compute_structure_features(self):
        """Step 5: Compute structure vectors for retrieval."""
        self.logger.info("🔧 Step 5: Computing Structure Features...")
        
        self.df = compute_structure_features_df(
            self.df,
            parsed_col=Config.PARSED_COL,
            output_col=Config.STRUCTURE_VECTOR_COL,
            wl_bins=Config.WL_FEATURE_SIZE,
            wl_iterations=Config.WL_ITERATIONS,
            max_vars=Config.MAX_VARIABLES,
            target_dim=Config.STRUCTURE_VECTOR_SIZE
        )
        
        # Statistics
        valid_vectors = self.df[Config.STRUCTURE_VECTOR_COL].apply(
            lambda x: isinstance(x, np.ndarray) and len(x) > 0
        ).sum()
        
        vector_dims = self.df[Config.STRUCTURE_VECTOR_COL].iloc[0].shape[0] if valid_vectors > 0 else 0
        
        self.stats['features'] = {
            'valid_vectors': int(valid_vectors),
            'vector_dimensions': int(vector_dims),
            'feature_success_rate': float(valid_vectors / len(self.df))
        }
        
        self.logger.info(f"   Generated structure vectors: {valid_vectors}/{len(self.df)}")
        self.logger.info(f"   Vector dimensions: {vector_dims}")
        
        if valid_vectors > 0:
            sample_vec = self.df[Config.STRUCTURE_VECTOR_COL].iloc[0]
            self.logger.info(f"   Sample vector stats: mean={sample_vec.mean():.4f}, std={sample_vec.std():.4f}")
    
    def _save_results(self):
        """Step 6: Save all preprocessing results."""
        self.logger.info("💾 Step 6: Saving Results...")
        
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the complete preprocessed DataFrame
        output_csv = f"{Config.OUTPUT_DIR}/metadata_{timestamp}.csv"
        self.df.to_csv(output_csv, index=False)
        self.logger.info(f"   Saved metadata to: {output_csv}")
        
        # Save structure vectors as numpy array
        if Config.STRUCTURE_VECTOR_COL in self.df.columns:
            vectors = np.stack(self.df[Config.STRUCTURE_VECTOR_COL].values)
            vectors_path = f"{Config.OUTPUT_DIR}/embeddings_{timestamp}.npy"
            np.save(vectors_path, vectors)
            self.logger.info(f"   Saved structure vectors to: {vectors_path}")
        
        # Save pipeline statistics
        stats_path = f"{Config.OUTPUT_DIR}/pipeline_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.logger.info(f"   Saved pipeline stats to: {stats_path}")
        
        # Save a simplified version for inspection
        simplified_cols = [
            Config.PROBLEM_ID_COL, Config.CLEANED_COL, Config.FINGERPRINT_COL,
            'problem_type', 'difficulty_level', 'equation_count', 'variable_count'
        ]
        simplified_df = self.df[simplified_cols].copy()
        simplified_path = f"{Config.OUTPUT_DIR}/preview_{timestamp}.csv"
        simplified_df.to_csv(simplified_path, index=False)
        self.logger.info(f"   Saved preview to: {simplified_path}")
        
        self.stats['output_files'] = {
            'metadata_csv': output_csv,
            'embeddings_npy': vectors_path,
            'stats_json': stats_path,
            'preview_csv': simplified_path
        }
    
    def _print_summary(self):
        """Print a comprehensive summary of the pipeline execution."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📈 PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*60)
        
        self.logger.info(f"📊 Dataset: {self.stats['original_count']} problems")
        self.logger.info(f"🧹 Cleaning: {self.stats['cleaning_report']['changed_texts']} texts modified")
        self.logger.info(f"🔍 Canonicalization: {self.stats['canonicalization']['success_rate']:.1%} success rate")
        self.logger.info(f"📈 Metadata: {len(self.stats['metadata']['problem_types'])} problem types")
        self.logger.info(f"🔧 Features: {self.stats['features']['valid_vectors']} structure vectors generated")
        
        if 'output_files' in self.stats:
            self.logger.info("💾 Output Files:")
            for file_type, file_path in self.stats['output_files'].items():
                self.logger.info(f"   - {file_type}: {file_path}")

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        pipeline = PreprocessPipeline()
        result_df = pipeline.run()
        
        print("\n🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📁 Output files saved in: {Config.OUTPUT_DIR}")
        print(f"📊 Processed {len(result_df)} problems")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()