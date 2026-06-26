import { useState } from "react";
import agg from "../../data/aggregates.json";
import s from "./PremiumChart.module.css";

type Stat = { mean: number; lo: number; hi: number };
const P = agg.premium_by_model_comm as Record<string, { off: Stat; on: Stat }>;
const MODELS = [
  { key: "haiku", short: "Haiku", color: "var(--d-haiku)" },
  { key: "gpt5mini", short: "GPT-5-mini", color: "var(--d-gpt)" },
  { key: "gemini", short: "Gemini", color: "var(--d-gemini)" },
];

const W = 520, H = 330, padT = 34, padB = 56;
const plotH = H - padT - padB;
const MAX = 34, MIN = -54, RANGE = MAX - MIN, scale = plotH / RANGE;
const zeroY = padT + MAX * scale;
const groupW = W / MODELS.length, bw = 46;

export function PremiumChart() {
  const [comm, setComm] = useState<"off" | "on">("off");
  const [hover, setHover] = useState<number | null>(null);

  return (
    <div className={s.wrap}>
      <div className={s.head}>
        <div className={s.toggle} role="group" aria-label="Communication condition">
          <button className={comm === "off" ? s.on : s.idle} onClick={() => setComm("off")}>COMM OFF</button>
          <button className={comm === "on" ? s.on : s.idle} onClick={() => setComm("on")}>COMM ON</button>
        </div>
        <p className={s.caption}>
          {comm === "off"
            ? "Without a voice, every strong model beats the weak Llama."
            : "Give them a voice - and every one is beaten by it."}
        </p>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className={s.svg} onMouseLeave={() => setHover(null)} role="img"
           aria-label="Capability premium by communication condition">
        {Array.from({ length: 13 }).map((_, c) =>
          Array.from({ length: 8 }).map((_, r) => (
            <circle key={`${c}-${r}`} cx={(c + 0.5) * (W / 13)} cy={padT + (r + 0.2) * (plotH / 8)} r={0.9} fill="var(--c-hairline)" />
          ))
        )}

        {MODELS.map((m, i) => {
          const st = P[m.key][comm];
          const v = st.mean;
          const h = Math.abs(v) * scale;
          const y = v >= 0 ? zeroY - h : zeroY;
          const cx = i * groupW + groupW / 2;
          const dimmed = hover !== null && hover !== i;
          return (
            <g
              key={m.key}
              className={dimmed ? s.dim : s.grp}
              onMouseEnter={() => setHover(i)}
            >
              <rect
                x={cx - bw / 2}
                width={bw}
                rx={2}
                fill={m.color}
                className={s.bar}
                style={{ y, height: h } as React.CSSProperties}
              />
              <text x={cx} y={H - 32} textAnchor="middle" className={s.xlabel}>{m.short}</text>
              <text x={cx} y={v >= 0 ? y - 10 : y + h + 22} textAnchor="middle" className={s.vlabel}>
                {v >= 0 ? `+${v.toFixed(0)}` : v.toFixed(0)}
              </text>
              {hover === i && (
                <text x={cx} y={v >= 0 ? y - 26 : y + h + 38} textAnchor="middle" className={s.callout}>
                  {`${v >= 0 ? "+" : ""}${v.toFixed(1)}  ·  95% CI [${st.lo.toFixed(0)}, ${st.hi.toFixed(0)}]`}
                </text>
              )}
            </g>
          );
        })}

        <line x1={8} x2={W - 8} y1={zeroY} y2={zeroY} className={s.zero} />
        <text x={W - 8} y={zeroY - 8} textAnchor="end" className={s.zerolbl}>premium = 0</text>
      </svg>
    </div>
  );
}
