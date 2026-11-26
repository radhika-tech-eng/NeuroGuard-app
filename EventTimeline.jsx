
export default function EventTimeline({ results = [], selectedWindow, onSelect }) {
  return (
    <section className="ng-panel">
      <div className="ng-panel-heading">
        <div>
          <h2>Event Timeline</h2>
          <p>Model-flagged windows · Research review</p>
        </div>
        <span>{results.length} windows</span>
      </div>

      {!results.length ? (
        <p>Run an EEG analysis to populate the timeline.</p>
      ) : (
        <>
          <div className="ng-timeline">
            {results.map((item) => {
              const score = item.model_scores?.ensemble ?? 0;
              const flagged = item.research_flag;

              return (
                <button
                  type="button"
                  key={item.window_index}
                  className={`ng-timeline-item ${
                    selectedWindow === item.window_index ? "selected" : ""
                  } ${flagged ? "flagged" : ""}`}
                  title={`Window ${item.window_index + 1}: score ${(
                    score * 100
                  ).toFixed(1)}%`}
                  onClick={() => onSelect?.(item.window_index)}
                >
                  <span className="ng-timeline-dot" />
                  <span className="ng-timeline-label">
                    {item.window_index + 1}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="ng-timeline-legend">
            <span><i className="normal-dot" /> Not flagged</span>
            <span><i className="flagged-dot" /> Flagged for review</span>
          </div>

          <div className="ng-event-list">
            {results
              .filter((item) => item.research_flag)
              .map((item) => (
                <button
                  type="button"
                  key={item.window_index}
                  onClick={() => onSelect?.(item.window_index)}
                >
                  <strong>Window {item.window_index + 1}</strong>
                  <span>
                    Ensemble score:{" "}
                    {(
                      (item.model_scores?.ensemble ?? 0) * 100
                    ).toFixed(1)}%
                  </span>
                  <span>Research flag — requires review</span>
                </button>
              ))}
          </div>
        </>
      )}
    </section>
  );
}
