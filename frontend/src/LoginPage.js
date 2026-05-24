import React, { useState } from "react";
import { login } from "./api";

function EyeIcon({ open }) {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {open ? (
        <>
          <path
            d="M2 12C3.8 8.5 7.4 6 12 6C16.6 6 20.2 8.5 22 12C20.2 15.5 16.6 18 12 18C7.4 18 3.8 15.5 2 12Z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.8" />
        </>
      ) : (
        <>
          <path
            d="M3 3L21 21"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          />
          <path
            d="M10.6 6.2C11.05 6.07 11.52 6 12 6C16.6 6 20.2 8.5 22 12C21.28 13.4 20.3 14.62 19.12 15.57M6.86 8.88C4.84 10.01 3.21 11.75 2 14C3.8 17.5 7.4 20 12 20C13.94 20 15.69 19.56 17.22 18.79M9.88 9.88C9.34 10.42 9 11.17 9 12C9 13.66 10.34 15 12 15C12.83 15 13.58 14.66 14.12 14.12"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </>
      )}
    </svg>
  );
}

function LoginPage({ onGoToSignup, onLoginSuccess }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (event) => {
    event.preventDefault();

    if (!username.trim() || !password.trim()) {
      setMessage("Please enter your username/email and password.");
      setIsError(true);
      return;
    }

    setLoading(true);
    try {
      const session = await login(username, password);
      setMessage("Login successful");
      setIsError(false);
      await onLoginSuccess(session);
    } catch (error) {
      setMessage(error.message || "Invalid credentials");
      setIsError(true);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.backgroundCircleOne}></div>
      <div style={styles.backgroundCircleTwo}></div>

      <div style={styles.card}>
        <div style={styles.logo}>AI</div>
        <h1 style={styles.title}>Welcome Back</h1>
        <p style={styles.subtitle}>Log in to AI Document Reader</p>

        <form onSubmit={handleLogin}>
          <label style={styles.label}>Username / Email</label>
          <input
            style={styles.input}
            type="text"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            placeholder="Enter your username"
          />

          <label style={styles.label}>Password</label>
          <div style={styles.passwordRow}>
            <input
              style={styles.passwordInput}
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter your password"
            />
            <button
              type="button"
              style={styles.eyeButton}
              onClick={() => setShowPassword((current) => !current)}
              aria-label={showPassword ? "Hide password" : "Show password"}
            >
              <EyeIcon open={showPassword} />
            </button>
          </div>

          <button style={styles.primaryButton} type="submit" disabled={loading}>
            {loading ? "Logging In..." : "Log In"}
          </button>
        </form>

        {message && (
          <div
            style={{
              ...styles.messageBox,
              backgroundColor: isError ? "#fef2f2" : "#eff6ff",
              color: isError ? "#b91c1c" : "#1d4ed8",
              border: isError ? "1px solid #fecaca" : "1px solid #bfdbfe",
            }}
          >
            {message}
          </div>
        )}

        <p style={styles.bottomText}>
          Don't have an account?{" "}
          <button type="button" style={styles.linkButton} onClick={onGoToSignup}>
            Sign Up
          </button>
        </p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #eef2ff 0%, #f8fafc 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "24px",
    position: "relative",
    overflow: "hidden",
    fontFamily: "Arial, sans-serif",
  },
  backgroundCircleOne: {
    position: "absolute",
    width: "320px",
    height: "320px",
    borderRadius: "50%",
    background: "rgba(59, 130, 246, 0.10)",
    top: "-80px",
    left: "-80px",
  },
  backgroundCircleTwo: {
    position: "absolute",
    width: "260px",
    height: "260px",
    borderRadius: "50%",
    background: "rgba(99, 102, 241, 0.10)",
    bottom: "-60px",
    right: "-60px",
  },
  card: {
    width: "100%",
    maxWidth: "430px",
    backgroundColor: "#ffffff",
    padding: "36px",
    borderRadius: "24px",
    boxShadow: "0 20px 50px rgba(15, 23, 42, 0.12)",
    position: "relative",
    zIndex: 1,
  },
  logo: {
    width: "64px",
    height: "64px",
    borderRadius: "16px",
    background: "linear-gradient(135deg, #2563eb, #4f46e5)",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontWeight: "bold",
    fontSize: "22px",
    margin: "0 auto 18px",
  },
  title: {
    margin: 0,
    textAlign: "center",
    fontSize: "34px",
    color: "#0f172a",
  },
  subtitle: {
    textAlign: "center",
    color: "#64748b",
    marginTop: "10px",
    marginBottom: "28px",
    fontSize: "15px",
  },
  label: {
    display: "block",
    marginBottom: "8px",
    marginTop: "14px",
    fontWeight: "bold",
    color: "#1e293b",
    fontSize: "14px",
  },
  input: {
    width: "100%",
    padding: "14px 16px",
    marginBottom: "8px",
    border: "1px solid #dbeafe",
    borderRadius: "12px",
    boxSizing: "border-box",
    fontSize: "15px",
    outline: "none",
    backgroundColor: "#f8fafc",
  },
  passwordRow: {
    position: "relative",
    marginBottom: "8px",
  },
  passwordInput: {
    width: "100%",
    padding: "14px 16px",
    paddingRight: "52px",
    border: "1px solid #dbeafe",
    borderRadius: "12px",
    boxSizing: "border-box",
    fontSize: "15px",
    outline: "none",
    backgroundColor: "#f8fafc",
  },
  eyeButton: {
    position: "absolute",
    top: "50%",
    right: "12px",
    transform: "translateY(-50%)",
    padding: "0",
    border: "none",
    backgroundColor: "transparent",
    color: "#64748b",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
  },
  primaryButton: {
    width: "100%",
    padding: "14px",
    background: "linear-gradient(135deg, #2563eb, #4f46e5)",
    color: "white",
    border: "none",
    borderRadius: "12px",
    cursor: "pointer",
    marginTop: "18px",
    fontWeight: "bold",
    fontSize: "15px",
    boxShadow: "0 10px 20px rgba(37, 99, 235, 0.25)",
  },
  messageBox: {
    marginTop: "18px",
    padding: "12px 14px",
    borderRadius: "12px",
    textAlign: "center",
    fontSize: "14px",
  },
  bottomText: {
    textAlign: "center",
    marginTop: "22px",
    color: "#475569",
    fontSize: "14px",
  },
  linkButton: {
    background: "none",
    border: "none",
    color: "#2563eb",
    cursor: "pointer",
    fontWeight: "bold",
    fontSize: "14px",
    padding: 0,
  },
};

export default LoginPage;
