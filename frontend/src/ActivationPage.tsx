import { useState } from "react";
import type { FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { activate } from "./hosted/api";

export default function ActivationPage({ token }: { token: string | null }) {
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [activated, setActivated] = useState(false);

  if (!token) {
    return (
      <main className="login-page">
        <Card className="login-card">
          <h1>Activation link unavailable</h1>
        </Card>
      </main>
    );
  }
  const activationToken = token;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (password.length < 12) {
      setError("Password must be at least 12 characters.");
      return;
    }
    if (password !== confirmation) {
      setError("Passwords must match.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      await activate(activationToken, password);
      setPassword("");
      setConfirmation("");
      setActivated(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Account activation failed.");
    } finally {
      setSubmitting(false);
    }
  }

  if (activated) {
    return (
      <main className="login-page">
        <Card className="login-card">
          <h1>Account activated</h1>
          <a href="/">Sign in</a>
        </Card>
      </main>
    );
  }

  return (
    <main className="login-page">
      <Card className="login-card">
        <form className="contents" onSubmit={(event) => { void handleSubmit(event); }}>
          <h1>Activate account</h1>
          {error && <div className="form-error" role="alert">{error}</div>}
          <label htmlFor="activation-password">Password</label>
          <Input
            id="activation-password"
            type="password"
            autoComplete="new-password"
            required
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
          <label htmlFor="activation-confirmation">Confirm password</label>
          <Input
            id="activation-confirmation"
            type="password"
            autoComplete="new-password"
            required
            value={confirmation}
            onChange={(event) => setConfirmation(event.target.value)}
          />
          <Button className="primary-button" type="submit" disabled={submitting}>
            {submitting ? "Activating…" : "Activate account"}
          </Button>
        </form>
      </Card>
    </main>
  );
}
