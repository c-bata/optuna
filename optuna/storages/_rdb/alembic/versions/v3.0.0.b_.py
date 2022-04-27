"""Add an enum type column into trial_intermediate_values table to represent +inf and -inf

Revision ID: v3.0.0.b
Revises: v3.0.0.a
Create Date: 2022-04-25 13:19:39.502964

"""
import enum

from alembic import op
import sqlalchemy as sa
import numpy as np

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm


# revision identifiers, used by Alembic.
revision = 'v3.0.0.b'
down_revision = 'v3.0.0.a'
branch_labels = None
depends_on = None


BaseModel = declarative_base()
RDB_MAX_FLOAT = np.finfo(np.float32).max
RDB_MIN_FLOAT = np.finfo(np.float32).min


class IntermediateValueModel(BaseModel):
    class FloatTypeEnum(enum.Enum):
        USE_VAL = 1  # Use the value of `intermediate_value` field.
        INF_POS = 2  # 'inf'
        INF_NEG = 3  # '-inf'

    __tablename__ = "trial_intermediate_values"
    trial_intermediate_value_id = sa.Column(sa.Integer, primary_key=True)
    intermediate_value = sa.Column(sa.Float, nullable=True)
    float_type = sa.Column(sa.Enum(FloatTypeEnum), nullable=False, default=FloatTypeEnum.USE_VAL)


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    with op.batch_alter_table('trial_intermediate_values', schema=None) as batch_op:
        batch_op.add_column(sa.Column('float_type', sa.Enum('USE_VAL', 'INF_POS', 'INF_NEG', name='floattypeenum'), nullable=False, default="USE_VAL"))
        batch_op.alter_column('intermediate_value',
               existing_type=sa.FLOAT(),
               nullable=True)

    try:
        records = session.query(IntermediateValueModel).all()
        mapping = []
        for r in records:
            float_type: IntermediateValueModel.FloatTypeEnum
            if np.isclose(r.intermediate_value, RDB_MIN_FLOAT):
                float_type = IntermediateValueModel.FloatTypeEnum.INF_NEG
            elif np.isclose(r.intermediate_value, RDB_MAX_FLOAT):
                float_type = IntermediateValueModel.FloatTypeEnum.INF_POS
            else:
                continue
            mapping.append({
                'trial_intermediate_value_id': r.trial_intermediate_value_id,
                'float_type': float_type
            })
        session.bulk_update_mappings(IntermediateValueModel, mapping)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise e
    finally:
        session.close()


def downgrade():
    with op.batch_alter_table('trial_intermediate_values', schema=None) as batch_op:
        batch_op.alter_column('intermediate_value',
               existing_type=sa.FLOAT(),
               nullable=False)
        batch_op.drop_column('float_type')
