import React from "react";
import ReactDOM from "react-dom/client";
import Rotunda from "./Rotunda";
import UnifiedHub from "./UnifiedHub";
import "./index.css";

const isLegacy = new URLSearchParams(window.location.search).get("legacy") === "1";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>{isLegacy ? <UnifiedHub /> : <Rotunda />}</React.StrictMode>
);
