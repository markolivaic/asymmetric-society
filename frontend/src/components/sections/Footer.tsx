import { Section } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import agg from "../../data/aggregates.json";
import s from "./Footer.module.css";

export function Footer() {
  return (
    <Section id="footer">
      <R>
        <p className={`display ${s.line}`}>
          Pre-registered. Reproducible.<br />
          Honest about the nulls.
        </p>
      </R>
      <R>
        <div className={s.meta}>
          <div className={s.col}>
            <p className={`mono ${s.label}`}>Study</p>
            <p className={s.val}>
              {agg.meta.n_games} games · {agg.meta.cells} balanced conditions ·
              €{agg.meta.spend_eur} total API spend
            </p>
          </div>
          <div className={s.col}>
            <p className={`mono ${s.label}`}>Links</p>
            <p className={s.val}>
              <a href="https://osf.io/rnqgs/">OSF pre-registration ↗</a>
            </p>
            <p className={s.val}>
              <a href="#">Source on GitHub ↗</a>
            </p>
          </div>
          <div className={s.col}>
            <p className={`mono ${s.label}`}>Author</p>
            <p className={s.val}>
              Francesco Marko Livaić
              <br />
              Mentor: prof. Aleksandar Stojanović
              <br />
              TVZ Zagreb · 2026
            </p>
          </div>
        </div>
      </R>
    </Section>
  );
}
