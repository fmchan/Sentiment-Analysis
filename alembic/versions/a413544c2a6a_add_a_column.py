"""Add a column

Revision ID: a413544c2a6a
Revises: 
Create Date: 2019-11-28 12:38:42.145821

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a413544c2a6a'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
  op.add_column('items', sa.Column('category', sa.String))

# sqlite doesnt support drop column...
def downgrade():
  op.drop_column('items', 'category')