import { useEffect, useRef, useState } from "react";

interface Props {
  title: string;
  message?: string;
  defaultValue?: string;
  placeholder?: string;
  confirmLabel?: string;
  cancelLabel?: string;
  validate?: (value: string) => string | null;
  onConfirm: (value: string) => void;
  onCancel: () => void;
}

export default function PromptDialog({
  title,
  message,
  defaultValue = "",
  placeholder,
  confirmLabel = "OK",
  cancelLabel = "Cancel",
  validate,
  onConfirm,
  onCancel,
}: Props) {
  const [value, setValue] = useState(defaultValue);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.select();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onCancel();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onCancel]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed) {
      setError("Value cannot be empty.");
      return;
    }
    if (validate) {
      const validationError = validate(trimmed);
      if (validationError) {
        setError(validationError);
        return;
      }
    }
    onConfirm(trimmed);
  };

  return (
    <div className="confirm-overlay" onClick={onCancel}>
      <div
        className="confirm-dialog"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="prompt-dialog-title"
      >
        <div id="prompt-dialog-title" className="confirm-dialog-title">
          {title}
        </div>
        {message && <div className="confirm-dialog-message">{message}</div>}
        <input
          ref={inputRef}
          type="text"
          aria-label={title}
          value={value}
          placeholder={placeholder}
          onChange={(event) => {
            setValue(event.target.value);
            setError(null);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              handleSubmit();
            }
          }}
          style={{
            width: "100%",
            padding: "8px 10px",
            fontSize: 13,
            border: `1px solid ${error ? "var(--app-error)" : "var(--app-border-strong)"}`,
            borderRadius: 6,
            marginBottom: error ? 4 : 16,
            boxSizing: "border-box",
          }}
        />
        {error && (
          <div
            style={{
              fontSize: 12,
              color: "var(--app-error)",
              marginBottom: 12,
            }}
          >
            {error}
          </div>
        )}
        <div className="confirm-dialog-actions">
          <button type="button" className="confirm-dialog-btn" onClick={onCancel}>
            {cancelLabel}
          </button>
          <button type="button" className="confirm-dialog-btn primary" onClick={handleSubmit}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
