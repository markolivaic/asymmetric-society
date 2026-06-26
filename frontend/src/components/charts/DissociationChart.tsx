import agg from "../../data/aggregates.json";
import s from "./DissociationChart.module.css";

type Stat = { mean: number; lo: number; hi: number };
const G = agg.premium_by_game as Record<string, { off: Stat; on: Stat }>;

const ROWS = [
  { label: "Public Goods · OFF", st: G.public_goods.off },
  { label: "Public Goods · ON", st: G.public_goods.on },
  { label: "Trust · OFF", st: G.trust_game.off },
  { label: "Trust · ON", st: G.trust_game.on },
];

const W = 560, rowH = 48, padT = 30, padB = 36, labelW = 150;
const H = padT + ROWS.length * rowH + padB;
const DMIN = -42, DMAX = 30;
const plotL = labelW, plotR = W - 18, plotW = plotR - plotL;
const x = (v: number) => plotL + ((v - DMIN) / (DMAX - DMIN)) * plotW;

export function DissociationChart() {
  const zeroX = x(0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={s.svg} role="img"
         aria-label="Strong-minus-weak premium by game, with 95% confidence intervals">
      <line x1={zeroX} x2={zeroX} y1={padT - 8} y2={H - padB + 6} className={s.zero} />
      <text x={zeroX} y={padT - 14} textAnchor="middle" className={s.zerolbl}>premium = 0</text>
      {[-40, -20, 0, 20].map((t) => (
        <text key={t} x={x(t)} y={H - 12} textAnchor="middle" className={s.tick}>{t > 0 ? `+${t}` : t}</text>
      ))}
      {ROWS.map((r, i) => {
        const y = padT + i * rowH + rowH / 2;
        const col = r.st.mean >= 0 ? "var(--d-strong)" : "var(--d-weak)";
        return (
          <g key={r.label}>
            <text x={labelW - 14} y={y + 4} textAnchor="end" className={s.rowlbl}>{r.label}</text>
            <line x1={x(r.st.lo)} x2={x(r.st.hi)} y1={y} y2={y} className={s.whisker} />
            <line x1={x(r.st.lo)} x2={x(r.st.lo)} y1={y - 5} y2={y + 5} className={s.cap} />
            <line x1={x(r.st.hi)} x2={x(r.st.hi)} y1={y - 5} y2={y + 5} className={s.cap} />
            <circle cx={x(r.st.mean)} cy={y} r={5.5} fill={col} stroke="var(--c-base)" strokeWidth={1.5} />
            <text x={x(r.st.mean)} y={y - 12} textAnchor="middle" className={s.val}>
              {r.st.mean >= 0 ? `+${r.st.mean.toFixed(0)}` : r.st.mean.toFixed(0)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
