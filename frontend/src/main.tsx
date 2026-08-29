import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createRootPage } from "./rootPage";
import "./App.css";

const page = createRootPage();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {page}
  </StrictMode>,
);
