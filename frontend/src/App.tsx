import { useLenis } from "./lib/useLenis";
import { Hook } from "./components/sections/Hook";
import { Setup } from "./components/sections/Setup";
import { Theater } from "./components/sections/Theater";
import { Reveal } from "./components/sections/Reveal";
import { ThesisCta } from "./components/sections/ThesisCta";
import { Footer } from "./components/sections/Footer";

export default function App() {
  useLenis();
  return (
    <>
      <Hook />
      <Setup />
      <Theater />
      <Reveal />
      <ThesisCta />
      <Footer />
    </>
  );
}
