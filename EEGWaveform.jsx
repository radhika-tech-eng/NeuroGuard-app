
import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function EEGWaveform({ eegData }) {
  const [channel, setChannel] = useState(0);

  const chartData = useMemo(() => {
    if (!eegData?.length) return [];

    // Display a maximum of 640 points for a smoother, lighter chart.
    const stride = Math.max(1, Math.ceil(eegData.length / 640));

    return eegData
      .filter((_, index) => index % stride === 0)
      .map((window) => ({
        sample: window.sample,
        amplitude: window.channels[channel],
      }));
  }, [eegData, channel]);

  return (
    <section className="ng-panel">
      <div className="ng-panel-heading">
        <div>
          <h2>EEG Waveform</h2>
          <p>Selected window · Channel-wise signal</p>
        </div>

        <select
          value={channel}
          onChange={(event) => setChannel(Number(event.target.value))}
          aria-label="Select EEG channel"
        >
          {Array.from({ length: 23 }, (_, index) => (
            <option key={index} value={index}>
              Channel {index + 1}
            </option>
          ))}
        </select>
      </div>

      {!chartData.length ? (
        <p>Upload an EEG file and run analysis to view the waveform.</p>
      ) : (
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263449" />
              <XAxis
                dataKey="sample"
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
              />
              <YAxis
                stroke="#94a3b8"
                tick={{ fontSize: 11 }}
                width={55}
              />
              <Tooltip
                contentStyle={{
                  background: "#101827",
                  border: "1px solid #334155",
                  borderRadius: 8,
                }}
              />
              <Line
                type="monotone"
                dataKey="amplitude"
                stroke="#4de2c5"
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
