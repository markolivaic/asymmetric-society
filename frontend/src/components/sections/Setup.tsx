import { Section, Kicker, StatChip } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import agg from "../../data/aggregates.json";
import s from "./Setup.module.css";

const BEATS = [
  {
    k: "01",
    t: "The players",
    d: "Three strong frontier models - Claude Haiku 4.5, GPT-5-mini, Gemini 2.5 Flash - rotate through the strong slots. The weak tier is a small local model, Llama 3.2 3B.",
  },
  {
    k: "02",
    t: "The games",
    d: "They play classic economic games - a Public Goods Game and a Trust Game - where cooperation pays, but free-riding pays more.",
  },
  {
    k: "03",
    t: "The twist",
    d: "Half the games ask each agent to make a public pledge before it acts - a message written in the belief the others will read it. They never do: the pledge is logged, never delivered. What flips the economy is being asked to commit - not being heard.",
  },
];

export function Setup() {
  return (
    <Section id="setup">
      <R>
        <Kicker>The setup</Kicker>
        <h2 className={`display ${s.title}`}>
          One economy.<br />
          Two kinds of mind.
        </h2>
        <p className={`lead ${s.lead}`}>
          We drop capable and limited models into the same games and watch the
          inequality emerge - not from rules we wrote, but from how they behave.
        </p>
      </R>

      <R>
        <ol className={s.beats}>
          {BEATS.map((b) => (
            <li key={b.k} className={s.beat}>
              <span className={`mono ${s.beatK}`}>{b.k}</span>
              <h3 className={s.beatT}>{b.t}</h3>
              <p className={s.beatD}>{b.d}</p>
            </li>
          ))}
        </ol>
      </R>

      <R>
        <div className={s.stats}>
          <StatChip value={agg.meta.n_games} label="games" />
          <StatChip value={3} label="strong models" />
          <StatChip value={agg.meta.cells} label="conditions" />
          <StatChip value={agg.meta.spend_eur} prefix="€" decimals={2} label="total spend" />
        </div>
      </R>
    </Section>
  );
}
