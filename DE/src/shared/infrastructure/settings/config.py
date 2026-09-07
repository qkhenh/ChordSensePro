"""
shared/infrastructure/settings/config.py

TODO: Bạn sẽ viết Settings class đọc config từ .env

Dùng pydantic-settings — tự động đọc biến môi trường + .env file.

Bạn cần viết:
1. class PostgresSettings(BaseSettings)
   - host, port, db, user, password
   - property url → trả về connection string

2. class MongoSettings(BaseSettings)
   - uri, db_name

3. class RedisSettings(BaseSettings)
   - url

4. class Settings(BaseSettings)
   - lồng 3 class trên vào
   - model_config = SettingsConfigDict(env_file=".env", ...)

5. Singleton pattern: get_settings() → cached Settings instance
"""
from __future__ import annotations
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class PostgresSettings(BaseSettings):
    host : str = "localhost"
    port : int = 5432
    db : str = "chordsense"
    user : str = "group26"
    password : str = "group26"
    
    model_config = SettingsConfigDict(env_prefix="POSTGRES_")
    
    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
    
class MongoSettings(BaseSettings):
    uri :     str = "mongodb://localhost:27017"
    db_name : str = "chordsense_raw"
    
    model_config = SettingsConfigDict(env_prefix="MONGO_")
    
class RedisSettings(BaseSettings):
    url : str = "redis://localhost:6379/0"
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
class Settings(BaseSettings):
    postgres : PostgresSettings = Field(default_factory=PostgresSettings)
    mongo : MongoSettings = Field(default_factory=MongoSettings)
    redis : RedisSettings = Field(default_factory=RedisSettings)
    
    app_env : str = "development"
    log_level : str = "INFO"
    data_dir : str = "./data"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()