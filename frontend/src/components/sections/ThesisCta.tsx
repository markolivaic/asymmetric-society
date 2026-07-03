import { Section, Kicker } from "../primitives";
import { Reveal as R } from "../../lib/Reveal";
import s from "./ThesisCta.module.css";

export function ThesisCta() {
  return (
    <Section id="thesis">
      <R>
        <div className={s.block}>
          <div>
            <Kicker>The write-up</Kicker>
            <h2 className={`display ${s.title}`}>Read the thesis.</h2>
            <p className={s.desc}>
              Fifty pages: the apparatus, the pre-registration, the results, and the
              honest fine print.
            </p>
          </div>
          <div className={s.action}>
            <a className={s.btn} href="/Asymmetric_Society.pdf" target="_blank" rel="noopener">
              Open the thesis (PDF) ↗
            </a>
            <p className={`mono ${s.meta}`}>PDF · 50 pages · 0.4 MB · English</p>
          </div>
        </div>
      </R>
    </Section>
  );
}
