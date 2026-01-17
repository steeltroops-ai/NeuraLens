"""Add retinal assessment tables

Revision ID: 20260115_001
Revises: 
Create Date: 2026-01-15 22:30:00.000000

This migration creates the core tables for the Retinal Analysis Pipeline:
- retinal_assessments: Stores all retinal biomarker data and risk assessments
- retinal_audit_logs: Audit trail for HIPAA compliance

Requirements: 8.1, 8.3, 8.4, 8.5, 8.6, 8.7, 8.11, 8.12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260115_001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table first if it doesn't exist (minimal version for FK support)
    # Note: This may already exist from initial setup
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_users'),
        sa.UniqueConstraint('email', name='uq_users_email')
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)
    
    # Create assessments table for general assessment sessions
    op.create_table(
        'assessments',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('modalities', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_assessments_user_id_users'),
        sa.PrimaryKeyConstraint('id', name='pk_assessments'),
        sa.UniqueConstraint('session_id', name='uq_assessments_session_id')
    )
    op.create_index('ix_assessments_session_id', 'assessments', ['session_id'], unique=True)
    
    # Create retinal_assessments table per design.md specification
    # Requirements: 8.4, 8.5, 8.6, 8.7
    op.create_table(
        'retinal_assessments',
        # Primary key - UUID format for uniqueness (Requirement 8.3)
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('patient_id', sa.String(length=255), nullable=False),
        
        # Image metadata (Requirement 8.5)
        sa.Column('original_image_url', sa.String(length=1024), nullable=False),
        sa.Column('processed_image_url', sa.String(length=1024), nullable=True),
        sa.Column('heatmap_url', sa.String(length=1024), nullable=True),
        sa.Column('segmentation_url', sa.String(length=1024), nullable=True),
        sa.Column('image_format', sa.String(length=50), nullable=True),
        sa.Column('image_resolution', sa.String(length=50), nullable=True),
        
        # Quality metrics (Requirement 8.5)
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('snr_db', sa.Float(), nullable=True),
        sa.Column('has_optic_disc', sa.Boolean(), nullable=True),
        sa.Column('has_macula', sa.Boolean(), nullable=True),
        
        # Biomarkers - Vessels (Requirement 8.4)
        sa.Column('vessel_density', sa.Float(), nullable=True),
        sa.Column('vessel_tortuosity', sa.Float(), nullable=True),
        sa.Column('avr_ratio', sa.Float(), nullable=True),
        sa.Column('branching_coefficient', sa.Float(), nullable=True),
        sa.Column('vessel_confidence', sa.Float(), nullable=True),
        
        # Biomarkers - Optic Disc (Requirement 8.4)
        sa.Column('cup_to_disc_ratio', sa.Float(), nullable=True),
        sa.Column('disc_area_mm2', sa.Float(), nullable=True),
        sa.Column('rim_area_mm2', sa.Float(), nullable=True),
        sa.Column('optic_disc_confidence', sa.Float(), nullable=True),
        
        # Biomarkers - Macula (Requirement 8.4)
        sa.Column('macular_thickness_um', sa.Float(), nullable=True),
        sa.Column('macular_volume_mm3', sa.Float(), nullable=True),
        sa.Column('macula_confidence', sa.Float(), nullable=True),
        
        # Biomarkers - Amyloid Beta (Requirement 8.4)
        sa.Column('amyloid_presence_score', sa.Float(), nullable=True),
        sa.Column('amyloid_distribution', sa.String(length=100), nullable=True),
        sa.Column('amyloid_confidence', sa.Float(), nullable=True),
        
        # Risk Assessment (Requirement 8.5)
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('risk_category', sa.String(length=50), nullable=False),
        sa.Column('confidence_lower', sa.Float(), nullable=True),
        sa.Column('confidence_upper', sa.Float(), nullable=True),
        
        # Processing metadata (Requirement 8.7)
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('processing_time_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True, default='completed'),
        
        # Timestamps (Requirement 8.6)
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        
        # Constraints
        sa.PrimaryKeyConstraint('id', name='pk_retinal_assessments')
    )
    
    # Create indexes for performance (Requirement 8.8 - <100ms retrieval)
    op.create_index('idx_patient_created', 'retinal_assessments', ['patient_id', 'created_at'])
    op.create_index('idx_user_created', 'retinal_assessments', ['user_id', 'created_at'])
    op.create_index('idx_risk_category', 'retinal_assessments', ['risk_category'])
    op.create_index('ix_retinal_assessments_patient_id', 'retinal_assessments', ['patient_id'])
    op.create_index('ix_retinal_assessments_user_id', 'retinal_assessments', ['user_id'])
    
    # Create retinal_audit_logs table per design.md specification
    # Requirements: 8.11, 8.12
    op.create_table(
        'retinal_audit_logs',
        sa.Column('id', sa.String(length=255), nullable=False),
        sa.Column('assessment_id', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),  # create, view, update, delete, export
        sa.Column('ip_address', sa.String(length=50), nullable=True),
        sa.Column('user_agent', sa.String(length=500), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        
        # Constraints
        sa.ForeignKeyConstraint(
            ['assessment_id'], 
            ['retinal_assessments.id'], 
            name='fk_retinal_audit_logs_assessment_id_retinal_assessments',
            ondelete='SET NULL'
        ),
        sa.PrimaryKeyConstraint('id', name='pk_retinal_audit_logs')
    )
    
    # Create indexes for audit log performance
    op.create_index('idx_assessment_timestamp', 'retinal_audit_logs', ['assessment_id', 'timestamp'])
    op.create_index('idx_user_timestamp', 'retinal_audit_logs', ['user_id', 'timestamp'])
    op.create_index('idx_audit_action', 'retinal_audit_logs', ['action'])
    
    # Create assessment_results table for modality-specific results
    op.create_table(
        'assessment_results',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('assessment_id', sa.Integer(), nullable=False),
        sa.Column('modality', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('biomarkers', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['assessment_id'], ['assessments.id'], name='fk_assessment_results_assessment_id_assessments'),
        sa.PrimaryKeyConstraint('id', name='pk_assessment_results')
    )
    
    # Create nri_results table for NRI fusion scores
    op.create_table(
        'nri_results',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('assessment_id', sa.Integer(), nullable=False),
        sa.Column('nri_score', sa.Float(), nullable=False),
        sa.Column('risk_category', sa.String(length=50), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('modality_contributions', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('target_value', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('environment', sa.String(length=50), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['assessment_id'], ['assessments.id'], name='fk_nri_results_assessment_id_assessments'),
        sa.PrimaryKeyConstraint('id', name='pk_nri_results')
    )
    
    # Create validation tables
    op.create_table(
        'validation_studies',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('name', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_validation_studies')
    )
    
    op.create_table(
        'validation_results',
        sa.Column('id', sa.Integer(), nullable=False, autoincrement=True),
        sa.Column('study_id', sa.Integer(), nullable=False),
        sa.Column('modality', sa.String(length=50), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('sensitivity', sa.Float(), nullable=True),
        sa.Column('specificity', sa.Float(), nullable=True),
        sa.Column('auc_roc', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('sample_size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['study_id'], ['validation_studies.id'], name='fk_validation_results_study_id_validation_studies'),
        sa.PrimaryKeyConstraint('id', name='pk_validation_results')
    )


def downgrade() -> None:
    # Drop tables in reverse order (respect foreign key constraints)
    op.drop_table('validation_results')
    op.drop_table('validation_studies')
    op.drop_table('nri_results')
    op.drop_table('assessment_results')
    
    op.drop_index('idx_audit_action', table_name='retinal_audit_logs')
    op.drop_index('idx_user_timestamp', table_name='retinal_audit_logs')
    op.drop_index('idx_assessment_timestamp', table_name='retinal_audit_logs')
    op.drop_table('retinal_audit_logs')
    
    op.drop_index('ix_retinal_assessments_user_id', table_name='retinal_assessments')
    op.drop_index('ix_retinal_assessments_patient_id', table_name='retinal_assessments')
    op.drop_index('idx_risk_category', table_name='retinal_assessments')
    op.drop_index('idx_user_created', table_name='retinal_assessments')
    op.drop_index('idx_patient_created', table_name='retinal_assessments')
    op.drop_table('retinal_assessments')
    
    op.drop_index('ix_assessments_session_id', table_name='assessments')
    op.drop_table('assessments')
    
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
