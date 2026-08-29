import App from "./App";
import ActivationPage from "./ActivationPage";

function captureActivationToken(): string | null {
  const match = /^#token=([A-Za-z0-9_-]+)$/.exec(window.location.hash);
  window.history.replaceState(
    window.history.state,
    "",
    window.location.pathname,
  );
  return match?.[1] ?? null;
}

export function createRootPage(): React.ReactNode {
  return window.location.pathname === "/activate"
    ? <ActivationPage token={captureActivationToken()} />
    : <App />;
}
