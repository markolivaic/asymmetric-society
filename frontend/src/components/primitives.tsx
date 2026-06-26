import type { ReactNode } from "react";
import { CountUp } from "../lib/CountUp";
import s from "./primitives.module.css";

/** Full-width band on the shared editorial grid. `dark` = the Theater contrast. */
export function Section({ id, dark, children }: { id?: string; dark?: boolean; children: ReactNode }) {
  return (
    <section id={id} className={`${s.band} ${dark ? s.dark : ""}`}>
      <div className={s.container}>{children}</div>
    </section>
  );
}

export function Kicker({ children }: { children: ReactNode }) {
  return <p className={s.kicker}>{children}</p>;
}

export function StatChip({ value, label, prefix = "", suffix = "", decimals = 0 }: {
  value: number;
  label: string;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}) {
  return (
    <div className={s.stat}>
      <span className={s.statN}>
        <CountUp value={value} prefix={prefix} suffix={suffix} decimals={decimals} />
      </span>
      <span className={s.statL}>{label}</span>
    </div>
  );
}
