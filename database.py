
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./smartcompute.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserTable(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    hashed_password = Column(String)
    organization = Column(String, nullable=True)
    created_at = Column(DateTime)

class TaskTable(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    workload_type = Column(String)
    params = Column(JSON)
    optimization_goal = Column(String)
    device_chosen = Column(String)
    status = Column(String)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime)
    completed_at = Column(DateTime, nullable=True)

class RunMetricsTable(Base):
    __tablename__ = "run_metrics"

    id = Column(String, primary_key=True, index=True)
    task_id = Column(String, index=True)
    cpu_ms = Column(Float, nullable=True)
    gpu_ms = Column(Float, nullable=True)
    fpga_ms = Column(Float, nullable=True)
    energy_joules_est = Column(Float, nullable=True)
    mem_gb = Column(Float, nullable=True)
    speedup_vs_cpu = Column(Float, nullable=True)
    timestamp = Column(DateTime)

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
