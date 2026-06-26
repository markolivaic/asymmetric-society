import React from "react";
import ReactDOM from "react-dom/client";
// Direction B fonts (self-hosted via fontsource); Direction A fonts load from <link> in index.html
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "lenis/dist/lenis.css";
import "./styles/tokens.css";
import "./styles/global.css";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
