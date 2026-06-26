import { Section, Kicker } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import { PremiumChart } from "../charts/PremiumChart";
import { DissociationChart } from "../charts/DissociationChart";
import s from "./Reveal.module.css";

// Findings as an argument, each figure its own titled moment.
// Final figure titles are pulled from the Confluence "Figure Titles" page at polish.
const FINDINGS = [
  {
    src: "f3_deception_surprisal_vs_kl.svg",
    kicker: "Deception · the registered method",
    line: "Contemporaneous surprisal cleanly separates the tiers - the strong's actions match their words, the weak's don't. Global policy-KL gives the opposite, artefactual answer: the registered negative control. (Exploratory - registered as such on OSF.)",
  },
  {
    src: "f4_message_classes_by_tier.svg",
    kicker: "What they actually say",
    line: "A blind classifier reads every message: the strong speak almost entirely in cooperative promises, while the weak hold every deceptive claim and threat.",
  },
  {
    src: "f6_tier_inference_roc.svg",
    kicker: "Telling them apart",
    line: "From behaviour alone - no model names, just how they play - a classifier separates strong from weak agents at AUC 0.97.",
  },
  {
    src: "f2_inequality_by_composition.svg",
    kicker: "Inequality · the modest part",
    line: "Welfare inequality (Gini) rises only modestly in mixed populations. The real signal is the premium and its flip, not a dramatic spread.",
  },
  {
    src: "f5_trust_token_transfer.svg",
    kicker: "Who pays whom",
    line: "In strong–weak Trust games, net token flow runs slightly toward the weaker agent.",
  },
];

export function Reveal() {
  return (
    <Section id="findings">
      <R>
        <Kicker>The findings</Kicker>
        <h2 className={`display ${s.title}`}>Talk turns the strong into prey.</h2>
        <p className={`lead ${s.lead}`}>
          Without communication the strong win. Give them a voice, and - in group games -
          every one of them is exploited, the same flip in all three frontier models.
          But it only happens in groups.
        </p>
      </R>

      <R>
        <div className={s.headline}>
          <div className={s.chartCol}>
            <PremiumChart />
          </div>
          <aside className={s.read}>
            <p className={`mono ${s.figlabel}`}>Figure 1 · capability premium × communication</p>
            <p className={s.readp}>
              Each strong model's cumulative payoff minus the weak Llama's, in the
              same games. Toggle communication and watch all three swing below zero.
            </p>
            <p className={s.readp}>Hover a bar for its 95% bootstrap CI. The bold line is zero.</p>
          </aside>
        </div>
      </R>

      <R>
        <div className={s.dissoc}>
          <p className={`mono ${s.beatlabel}`}>It&rsquo;s the game, not the model</p>
          <p className={s.dissocLead}>
            The exploitation isn&rsquo;t universal - it&rsquo;s structural. In group games
            (Public Goods), where you can free-ride on everyone&rsquo;s public pledges, the
            premium swings hard: the strong win without communication, and are eaten alive
            with it. One-on-one in the Trust game it all but disappears - the strong never
            really pull ahead, and the talk-driven effect is weak and uncertain (look at the
            intervals). The exploitation is a property of the <em>group</em>, not the model.
          </p>
          <div className={s.dissocChartWrap}>
            <DissociationChart />
          </div>
          <p className={s.cap}>
            Strong-minus-weak payoff by game, 95% bootstrap CIs · OFF/ON = communication.
            Public Goods is large and tight; Trust sits near zero with wide, uncommitted intervals.
          </p>
        </div>
      </R>

      <div className={s.figFlow}>
        {FINDINGS.map((f) => (
          <R key={f.src}>
            <div className={s.figBlock}>
              <p className={`mono ${s.figKicker}`}>{f.kicker}</p>
              <p className={s.figLine}>{f.line}</p>
              <figure className={s.figFrame}>
                <img src={`/figures/${f.src}`} alt={f.line} loading="lazy" />
              </figure>
            </div>
          </R>
        ))}
      </div>

      <R>
        <div className={s.finePrint}>
          <p className={`mono ${s.beatlabel}`}>Where it breaks - the honest fine print</p>
          <ul className={s.fpList}>
            <li>
              <strong>Inequality is modest.</strong> The Gini barely separates the
              conditions; the real signal is the premium and its flip, not a dramatic spread.
            </li>
            <li>
              <strong>Trust shows no clear premium.</strong> The exploitation is specific to
              group free-riding, not one-on-one trust - see the dissociation above.
            </li>
            <li>
              <strong>The deception result is exploratory.</strong> Surprisal cleanly
              separates the tiers and the blind classifier agrees - but it is registered as
              exploratory on OSF: a methodological contribution, not a confirmed law.
            </li>
          </ul>
        </div>
      </R>
    </Section>
  );
}
