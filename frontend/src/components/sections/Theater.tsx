import { useEffect, useRef, useState } from "react";
import { Section, Kicker } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import games from "../../data/games.json";
import s from "./Theater.module.css";

const MODEL_LABEL: Record<string, string> = {
  haiku: "Claude Haiku 4.5",
  gpt5mini: "GPT-5-mini",
  gemini: "Gemini 2.5 Flash",
  llama: "Llama 3.2 3B",
};

const onGame = games.find((g) => g.comm && g.comp === "mixed") ?? games[0];
const offGame = games.find((g) => !g.comm && g.comp === "mixed") ?? onGame;

export function Theater() {
  const [comm, setComm] = useState(true);
  const [r, setR] = useState(0);
  const [playing, setPlaying] = useState(false);

  const game = comm ? onGame : offGame;
  const n = game.rounds.length;
  const ri = Math.min(r, n - 1);
  const round = game.rounds[ri];
  const byAgent = Object.fromEntries(round.actions.map((a) => [a.agent, a]));
  const maxCum = Math.max(...round.actions.map((a) => a.cumulative ?? 0), 1);

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => setR((x) => x + 1), 1700);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [playing]);
  useEffect(() => {
    if (ri >= n - 1) setPlaying(false);
  }, [ri, n]);

  return (
    <Section id="theater" dark>
      <R>
        <Kicker>Watch one game</Kicker>
        <h2 className={`display ${s.title}`}>See it play out.</h2>
        <p className={`lead ${s.lead}`}>
          The same four players, the same ten rounds. Flip communication on or off
          and scrub through - watch the strong pledge cooperation, and the weak cash in.
          One thing to know: the pledges you&rsquo;ll read were never delivered - each
          agent saw only the others&rsquo; actions.
        </p>
      </R>

      <R>
        <div className={s.stage}>
          <div className={s.bar}>
            <div className={s.toggle} role="group" aria-label="Communication condition">
              <button className={comm ? s.tOn : s.tIdle} onClick={() => setComm(true)}>● COMM ON</button>
              <button className={!comm ? s.tOn : s.tIdle} onClick={() => setComm(false)}>COMM OFF</button>
            </div>
            <span className={`mono ${s.roundlbl}`}>
              Round {String(ri + 1).padStart(2, "0")} / {n} · Public Goods · mixed
            </span>
          </div>

          <div className={s.scrub}>
            <button className={s.nav} onClick={() => setR((x) => Math.max(0, x - 1))} aria-label="Previous round">‹</button>
            <input
              className={s.range}
              type="range"
              min={0}
              max={n - 1}
              value={ri}
              onChange={(e) => { setPlaying(false); setR(+e.target.value); }}
              aria-label="Round"
            />
            <button className={s.nav} onClick={() => setR((x) => Math.min(n - 1, x + 1))} aria-label="Next round">›</button>
            <button className={s.play} onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
              {playing ? "❚❚" : "▶"}
            </button>
          </div>

          <ul className={s.agents} key={`${comm}-${ri}`}>
            {game.agents.map((ag) => {
              const a = byAgent[ag.id];
              return (
                <li key={ag.id} className={s.agent}>
                  <div className={s.who}>
                    <span className={s.dot} data-tier={ag.tier} />
                    <span className={`mono ${s.model}`}>{MODEL_LABEL[ag.model] ?? ag.model}</span>
                  </div>
                  <p className={s.msg}>
                    {comm && a?.message ? (
                      `“${a.message}”`
                    ) : (
                      <span className={s.silent}>{comm ? "-" : "no message - actions only"}</span>
                    )}
                  </p>
                  <div className={s.contrib}>
                    <span className={s.cn}>{a?.contribution ?? "–"}</span>
                    <span className={`mono ${s.cl}`}>of 20</span>
                  </div>
                </li>
              );
            })}
          </ul>

          <div className={s.standings}>
            <p className={`mono ${s.stlabel}`}>Standings · cumulative payoff after round {ri + 1}</p>
            {game.agents.map((ag) => {
              const cum = byAgent[ag.id]?.cumulative ?? 0;
              return (
                <div key={ag.id} className={s.strow}>
                  <span className={`mono ${s.stname}`}>{MODEL_LABEL[ag.model] ?? ag.model}</span>
                  <div className={s.sttrack}>
                    <div className={s.stfill} data-tier={ag.tier} style={{ width: `${(cum / maxCum) * 100}%` }} />
                  </div>
                  <span className={`mono ${s.stval}`}>{cum.toFixed(0)}</span>
                </div>
              );
            })}
          </div>
          <p className={`mono ${s.caveat}`}>
            One matched game shown - the population-level flip is the chart under “The findings”.
            Messages are visible to you, not to the players: logged by the runner, never passed on.
          </p>
        </div>
      </R>
    </Section>
  );
}
