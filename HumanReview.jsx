import { useCallback, useEffect, useState } from 'react';
import './HumanReview.css';
import './HumanReview.css';

const API_URL = 'http://127.0.0.1:8000';
const ANNOTATIONS = ['Possible seizure', 'Noise detected', 'Needs review'];

/**
 * Props:
 * predictionId: stable identifier for this prediction/window (required)
 * reviewerName: current reviewer name (required)
 */
export default function HumanReview({ predictionId, reviewerName }) {
  const [annotation, setAnnotation] = useState('Needs review');
  const [notes, setNotes] = useState('');
  const [status, setStatus] = useState('Reviewed');
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const loadHistory = useCallback(async () => {
    if (!predictionId) return;
    try {
      const response = await fetch(`${API_URL}/reviews/${encodeURIComponent(predictionId)}`);
      if (!response.ok) throw new Error('Could not load review history.');
      setHistory(await response.json());
    } catch (error) {
      setMessage(error.message);
    }
  }, [predictionId]);

  useEffect(() => { loadHistory(); }, [loadHistory]);

  async function saveReview(event) {
    event.preventDefault();
    if (!predictionId) {
      setMessage('Prediction ID is missing.');
      return;
    }
    if (!reviewerName?.trim()) {
      setMessage('Reviewer name is required.');
      return;
    }
    setBusy(true);
    setMessage('');
    try {
      const response = await fetch(`${API_URL}/reviews`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_id: String(predictionId),
          reviewer_name: reviewerName.trim(),
          annotation,
          notes,
          status,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not save review.');
      setMessage('Review saved successfully.');
      setNotes('');
      await loadHistory();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ng-human-review">
      <div className="ng-human-review__heading">
        <div>
          <span className="ng-human-review__eyebrow">HUMAN VALIDATION</span>
          <h3>Review &amp; Annotation</h3>
          <p>Record a reviewer assessment for this prediction.</p>
        </div>
        <span className="ng-human-review__status">{history.length ? `${history.length} review(s)` : 'Not reviewed yet'}</span>
      </div>

      <form onSubmit={saveReview} className="ng-human-review__form">
        <label htmlFor="ng-review-annotation">Annotation</label>
        <select id="ng-review-annotation" value={annotation} onChange={e => setAnnotation(e.target.value)}>
          {ANNOTATIONS.map(item => <option key={item} value={item}>{item}</option>)}
        </select>

        <label htmlFor="ng-review-notes">Reviewer notes</label>
        <textarea id="ng-review-notes" value={notes} onChange={e => setNotes(e.target.value)}
          placeholder="Add observations about this EEG window..." rows={3} />

        <label htmlFor="ng-review-status">Reviewer status</label>
        <select id="ng-review-status" value={status} onChange={e => setStatus(e.target.value)}>
          <option>Reviewed</option>
          <option>Needs follow-up</option>
        </select>

        <button type="submit" disabled={busy || !predictionId || !reviewerName?.trim()}>
          {busy ? 'Saving...' : '✓ Mark as Reviewed & Save'}
        </button>
        {message && <p className="ng-human-review__message" role="status">{message}</p>}
      </form>

      <div className="ng-human-review__history">
        <h4>Review History</h4>
        {history.length === 0 ? <p>No saved reviews for this prediction.</p> : history.map(item => (
          <article key={item.review_id} className="ng-human-review__record">
            <div><strong>{item.annotation}</strong><span>{item.status}</span></div>
            <p>{item.notes || 'No additional notes.'}</p>
            <small>By {item.reviewer_name} · {new Date(item.created_at).toLocaleString()}</small>
          </article>
        ))}
      </div>
    </section>
  );
}
