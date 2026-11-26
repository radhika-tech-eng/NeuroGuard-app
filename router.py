from fastapi import APIRouter, HTTPException, Query
from .database import get_connection, initialize_database
from .schemas import ReviewCreate, ReviewUpdate

router = APIRouter(prefix='/reviews', tags=['Human Review'])


@router.on_event('startup')
def setup_review_database():
    initialize_database()


@router.post('', status_code=201)
def create_review(payload: ReviewCreate):
    with get_connection() as connection:
        cursor = connection.execute(
            '''INSERT INTO human_reviews
               (prediction_id, reviewer_name, annotation, notes, status)
               VALUES (?, ?, ?, ?, ?)''',
            (payload.prediction_id, payload.reviewer_name.strip(), payload.annotation,
             payload.notes, payload.status),
        )
        connection.commit()
        row = connection.execute(
            'SELECT * FROM human_reviews WHERE review_id = ?', (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


@router.get('/history')
def review_history(limit: int = Query(default=100, ge=1, le=500)):
    with get_connection() as connection:
        rows = connection.execute(
            'SELECT * FROM human_reviews ORDER BY review_id DESC LIMIT ?', (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


@router.get('/{prediction_id}')
def reviews_for_prediction(prediction_id: str):
    with get_connection() as connection:
        rows = connection.execute(
            'SELECT * FROM human_reviews WHERE prediction_id = ? ORDER BY review_id DESC',
            (prediction_id,),
        ).fetchall()
        return [dict(row) for row in rows]


@router.patch('/item/{review_id}')
def update_review(review_id: int, payload: ReviewUpdate):
    changes = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail='No review fields supplied.')
    allowed = {'annotation', 'notes', 'status'}
    if not set(changes).issubset(allowed):
        raise HTTPException(status_code=400, detail='Invalid review fields.')
    assignments = ', '.join(f'{field} = ?' for field in changes)
    values = list(changes.values()) + [review_id]
    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE human_reviews SET {assignments}, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE review_id = ?",
            values,
        )
        connection.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail='Review not found.')
        row = connection.execute(
            'SELECT * FROM human_reviews WHERE review_id = ?', (review_id,)
        ).fetchone()
        return dict(row)
