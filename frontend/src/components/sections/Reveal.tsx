import { Section, Kicker } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import { PremiumChart } from "../charts/PremiumChart";
import { DissociationChart } from "../charts/DissociationChart";
import s from "./Reveal.module.css";

// Findings as an argument, each figure its own titled moment.
// Titles are the final ones from the figure-title registry (thesis + frontend).
const FINDINGS = [
  {
    src: "f3_deception_surprisal_vs_kl.svg",
    kicker: "Deception · exploratory",
    title: "Catching Lies Where Policy-Divergence Fails",
    line: "Contemporaneous surprisal cleanly separates the tiers - the strong's actions match their words, the weak's don't. Global policy-KL gives the opposite, artefactual answer: the registered negative control. (Exploratory - registered as such on OSF.)",
  },
  {
    src: "f4_message_classes_by_tier.svg",
    kicker: "What they say",
    title: "The Strong Promise, the Weak Deceive",
    line: "A blind classifier reads every message: the strong speak almost entirely in cooperative promises, while the weak hold every deceptive claim and threat.",
  },
  {
    src: "f6_tier_inference_roc.svg",
    kicker: "Tier inference",
    title: "Behavior Betrays Capability",
    line: "From behaviour alone - no model names, just how they play - a classifier separates strong from weak agents at AUC 0.97.",
  },
  {
    src: "f2_inequality_by_composition.svg",
    kicker: "Inequality",
    title: "Mixing Capabilities Breeds Inequality",
    line: "Welfare inequality (Gini) rises only modestly in mixed populations. The real signal is the premium and its flip, not a dramatic spread.",
  },
  {
    src: "f5_trust_token_transfer.svg",
    kicker: "Trust transfers",
    title: "Trust Flows Downhill to the Weak",
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
          Without communication the strong win. Ask them to promise in public, and - in
          group games - every one of them is exploited, the same flip in all three
          frontier models. No promise is ever delivered. It only happens in groups.
        </p>
      </R>

      <R>
        <div className={s.headline}>
          <div className={s.chartCol}>
            <PremiumChart />
          </div>
          <aside className={s.read}>
            <p className={`mono ${s.figlabel}`}>Figure 1 · capability premium × communication</p>
            <h3 className={`display ${s.figTitle}`}>Talk Makes the Strong Weak</h3>
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
          <h3 className={`display ${s.figTitle}`}>Public Goods Inverts, Trust Stays Flat</h3>
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
            The communication-driven inversion is specific to the Public Goods Game: the
            premium swings from +22 to −31 cumulative tokens when communication is on, while
            in the Trust Game it stays near zero (−4 to −16) with wide, uncertain intervals
            in both conditions. 95% bootstrap CIs.
          </p>
        </div>
      </R>

      <div className={s.figFlow}>
        {FINDINGS.map((f) => (
          <R key={f.src}>
            <div className={s.figBlock}>
              <p className={`mono ${s.figKicker}`}>{f.kicker}</p>
              <h3 className={`display ${s.figTitle}`}>{f.title}</h3>
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
              <strong>No message is ever delivered.</strong> Each agent is asked to pledge
              in public and told the others will see it - but the runner never passes
              messages on; agents act on past actions alone. The flip is the framing of
              public commitment, not an exchange of cheap talk.
            </li>
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
