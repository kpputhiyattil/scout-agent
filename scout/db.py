"""SQLAlchemy models + session helpers. Raw pipeline output is immutable;
coach corrections live in Override and are applied as a join at read time."""
from __future__ import annotations

import datetime as dt
import json

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from scout.config import get_settings


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)            # URL or original filename
    video_path: Mapped[str] = mapped_column(String)
    fps: Mapped[float] = mapped_column(Float, default=25.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)
    players: Mapped[list[Player]] = relationship(back_populates="match")


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    track_id: Mapped[int] = mapped_column(Integer)
    team: Mapped[str] = mapped_column(String, default="?")       # A | B | ref
    jersey: Mapped[int | None] = mapped_column(Integer, nullable=True)
    name: Mapped[str] = mapped_column(String, default="Unknown")
    role: Mapped[str] = mapped_column(String, default="?")       # GK|DEF|MID|ATT
    role_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    minutes: Mapped[float] = mapped_column(Float, default=0.0)
    match: Mapped[Match] = relationship(back_populates="players")


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    t: Mapped[float] = mapped_column(Float)                      # seconds
    type: Mapped[str] = mapped_column(String)                    # pass|shot|duel|save|...
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    success: Mapped[int] = mapped_column(Integer, default=1)
    x: Mapped[float] = mapped_column(Float, default=0.0)         # pitch meters
    y: Mapped[float] = mapped_column(Float, default=0.0)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class Rating(Base):
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    role: Mapped[str] = mapped_column(String)
    overall: Mapped[float] = mapped_column(Float)
    sub_scores: Mapped[dict] = mapped_column(JSON, default=dict)   # group -> 0-100
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)     # kpi -> raw value
    note: Mapped[str] = mapped_column(Text, default="")            # LLM scouting note


class Override(Base):
    """Coach corrections. field in {team, jersey, name, role}."""
    __tablename__ = "overrides"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    field: Mapped[str] = mapped_column(String)
    value: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class JobStatus(Base):
    __tablename__ = "job_status"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(String)
    stage: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|running|done|failed
    detail: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow,
                                                    onupdate=dt.datetime.utcnow)


_engine = None


def get_engine(url: str | None = None):
    global _engine
    if _engine is None:
        _engine = create_engine(url or get_settings().db_url)
        Base.metadata.create_all(_engine)
    return _engine


def get_session(url: str | None = None) -> Session:
    return Session(get_engine(url))


def apply_overrides(session: Session, player: Player) -> Player:
    """Return player with coach overrides applied (latest wins). Does not mutate DB."""
    rows = (session.query(Override).filter_by(player_id=player.id)
            .order_by(Override.created_at).all())
    for o in rows:
        if o.field == "jersey":
            player.jersey = int(o.value)
        elif o.field in ("team", "name", "role"):
            setattr(player, o.field, o.value)
    return player
