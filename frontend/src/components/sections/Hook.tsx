import { Kicker } from "../primitives";
import s from "./Hook.module.css";

export function Hook() {
  return (
    <header className={s.hero}>
      <div className={s.inner}>
        <span className={s.rule} aria-hidden />
        <div className={s.content}>
          <Kicker>Bachelor thesis · TVZ Zagreb · 2026</Kicker>
          <h1 className={`display ${s.title}`}>
            What happens when cooperative&nbsp;AI<br />
            meets AI that isn&rsquo;t?
          </h1>
          <p className={`lead ${s.lead}`}>
            Strong frontier models and weak local models share one economy.
            We measure who exploits whom - and the answer flips the moment they can talk.
          </p>
          <a className={s.cta} href="#findings">
            Read the findings <span>&rarr;</span>
          </a>
        </div>
      </div>
    </header>
  );
}
