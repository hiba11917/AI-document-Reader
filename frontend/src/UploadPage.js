import React, { useMemo, useState } from "react";
import { uploadDocument } from "./api";

const MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = [".pdf", ".doc", ".docx", ".xls", ".xlsx"];

function formatFileSize(sizeInBytes) {
  if (!sizeInBytes && sizeInBytes !== 0) {
    return "";
  }
  if (sizeInBytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(sizeInBytes / 1024))} KB`;
  }
  return `${(sizeInBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileExtension(filename = "") {
  const dotIndex = filename.lastIndexOf(".");
  if (dotIndex === -1) {
    return "";
  }
  return filename.slice(dotIndex).toLowerCase();
}

function validateSelectedFile(file) {
  if (!file) {
    return "Please select a document first.";
  }

  const extension = getFileExtension(file.name);
  if (!SUPPORTED_EXTENSIONS.includes(extension)) {
    return "Unsupported file type. Please upload a PDF, DOC, DOCX, XLS, or XLSX file.";
  }

  if (file.size <= 0) {
    return "The selected file is empty. Please choose a file with content.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return "File is too large. Please choose a file smaller than 200 MB.";
  }

  return "";
}

function UploadPage({ onBackToDashboard, onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [statusTone, setStatusTone] = useState("info");
  const [isUploading, setIsUploading] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [validationTips, setValidationTips] = useState([]);

  const uploadSteps = useMemo(
    () => [
      "Uploading document to the server...",
      "Extracting text and preparing pages...",
      "Creating embeddings and indexing for search...",
      "Saving document metadata and storage record...",
    ],
    []
  );

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file || null);
    setCurrentStepIndex(-1);

    if (!file) {
      setUploadStatus("");
      setStatusTone("info");
      setValidationTips([]);
      return;
    }

    const validationError = validateSelectedFile(file);
    if (validationError) {
      setUploadStatus(validationError);
      setStatusTone("error");
    } else {
      setUploadStatus(
        `"${file.name}" is ready to upload. File size: ${formatFileSize(file.size)}.`
      );
      setStatusTone("info");
    }

    setValidationTips([
      `File type: ${getFileExtension(file.name).replace(".", "").toUpperCase() || "Unknown"}`,
      `File size: ${formatFileSize(file.size)}`,
      "Maximum allowed size: 200 MB",
    ]);
  };

  const handleUpload = async () => {
    const validationError = validateSelectedFile(selectedFile);
    if (validationError) {
      setUploadStatus(validationError);
      setStatusTone("error");
      return;
    }

    setIsUploading(true);
    setUploadStatus(uploadSteps[0]);
    setStatusTone("info");
    setCurrentStepIndex(0);

    let progressTimer = null;
    try {
      progressTimer = window.setInterval(() => {
        setCurrentStepIndex((currentIndex) => {
          const nextIndex = Math.min(currentIndex + 1, uploadSteps.length - 1);
          setUploadStatus(uploadSteps[nextIndex]);
          return nextIndex;
        });
      }, 1800);

      const document = await uploadDocument(selectedFile);

      if (progressTimer) {
        window.clearInterval(progressTimer);
      }

      setUploadStatus(
        `"${document.filename}" uploaded, indexed, and saved to storage successfully.`
      );
      setStatusTone("success");
      setCurrentStepIndex(uploadSteps.length - 1);
      onUploadSuccess(document);
    } catch (error) {
      if (progressTimer) {
        window.clearInterval(progressTimer);
      }

      setUploadStatus(
        error.message ||
          "Upload failed. Please check the file type, file size, and backend connection."
      );
      setStatusTone("error");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.card}>
          <div style={styles.iconBox}>^</div>

          <h1 style={styles.title}>Upload Document</h1>
          <p style={styles.subtitle}>
            Select a file to upload into the AI Document Reader
          </p>

          <div style={styles.uploadBox}>
            <input
              type="file"
              onChange={handleFileChange}
              style={styles.fileInput}
              accept=".pdf,.doc,.docx,.xls,.xlsx"
            />

            {selectedFile ? (
              <div>
                <p style={styles.fileName}>
                  Selected file: <strong>{selectedFile.name}</strong>
                </p>
                <p style={styles.fileMeta}>
                  {getFileExtension(selectedFile.name).toUpperCase().replace(".", "")} file
                  {" • "}
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            ) : (
              <p style={styles.placeholderText}>No document selected yet.</p>
            )}
          </div>

          <div style={styles.buttonRow}>
            <button
              style={{
                ...styles.uploadButton,
                ...(isUploading ? styles.disabledButton : null),
              }}
              onClick={handleUpload}
              disabled={isUploading}
            >
              {isUploading ? "Processing..." : "Upload Document"}
            </button>

            <button style={styles.backButton} onClick={onBackToDashboard}>
              Back to Dashboard
            </button>
          </div>

          {uploadStatus && (
            <div
              style={{
                ...styles.statusBox,
                ...(statusTone === "success" ? styles.successBox : null),
                ...(statusTone === "warning" ? styles.warningBox : null),
                ...(statusTone === "error" ? styles.errorBox : null),
              }}
            >
              {uploadStatus}
            </div>
          )}

          {validationTips.length > 0 && (
            <div style={styles.validationCard}>
              {validationTips.map((tip) => (
                <p key={tip} style={styles.validationText}>
                  {tip}
                </p>
              ))}
            </div>
          )}

          {isUploading && (
            <div style={styles.stepsCard}>
              {uploadSteps.map((step, index) => (
                <div key={step} style={styles.stepRow}>
                  <span
                    style={{
                      ...styles.stepDot,
                      ...(index <= currentStepIndex ? styles.stepDotActive : null),
                    }}
                  />
                  <span
                    style={{
                      ...styles.stepText,
                      ...(index === currentStepIndex ? styles.stepTextActive : null),
                    }}
                  >
                    {step}
                  </span>
                </div>
              ))}
            </div>
          )}

          <div style={styles.infoCard}>
            <p style={styles.infoTitle}>Supported document types</p>
            <p style={styles.infoText}>PDF, DOC, DOCX, XLS, XLSX</p>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(180deg, #eef2ff 0%, #f8fafc 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    padding: "24px",
    fontFamily: "Arial, sans-serif",
  },
  container: {
    width: "100%",
    maxWidth: "680px",
  },
  card: {
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    padding: "36px",
    boxShadow: "0 20px 50px rgba(15, 23, 42, 0.10)",
    textAlign: "center",
  },
  iconBox: {
    width: "72px",
    height: "72px",
    margin: "0 auto 18px",
    borderRadius: "20px",
    background: "linear-gradient(135deg, #2563eb, #4f46e5)",
    color: "#fff",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "32px",
    fontWeight: "bold",
  },
  title: {
    margin: 0,
    fontSize: "34px",
    color: "#0f172a",
  },
  subtitle: {
    marginTop: "10px",
    marginBottom: "28px",
    color: "#64748b",
    fontSize: "15px",
  },
  uploadBox: {
    border: "2px dashed #cbd5e1",
    borderRadius: "18px",
    padding: "28px",
    backgroundColor: "#f8fafc",
    marginBottom: "22px",
  },
  fileInput: {
    marginBottom: "16px",
  },
  placeholderText: {
    color: "#64748b",
    margin: 0,
  },
  fileName: {
    color: "#0f172a",
    margin: 0,
  },
  fileMeta: {
    color: "#64748b",
    margin: "6px 0 0 0",
    fontSize: "13px",
  },
  buttonRow: {
    display: "flex",
    justifyContent: "center",
    gap: "12px",
    flexWrap: "wrap",
    marginBottom: "20px",
  },
  uploadButton: {
    padding: "14px 18px",
    border: "none",
    borderRadius: "12px",
    background: "linear-gradient(135deg, #2563eb, #4f46e5)",
    color: "#fff",
    fontWeight: "bold",
    cursor: "pointer",
    minWidth: "170px",
  },
  backButton: {
    padding: "14px 18px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#e2e8f0",
    color: "#0f172a",
    fontWeight: "bold",
    cursor: "pointer",
    minWidth: "170px",
  },
  disabledButton: {
    opacity: 0.7,
    cursor: "not-allowed",
  },
  statusBox: {
    marginTop: "10px",
    padding: "14px",
    borderRadius: "12px",
    fontSize: "14px",
    fontWeight: "bold",
    backgroundColor: "#eff6ff",
    color: "#1d4ed8",
    border: "1px solid #bfdbfe",
  },
  successBox: {
    backgroundColor: "#ecfdf5",
    color: "#166534",
    border: "1px solid #bbf7d0",
  },
  warningBox: {
    backgroundColor: "#fff7ed",
    color: "#c2410c",
    border: "1px solid #fdba74",
  },
  errorBox: {
    backgroundColor: "#fef2f2",
    color: "#b91c1c",
    border: "1px solid #fecaca",
  },
  validationCard: {
    marginTop: "16px",
    textAlign: "left",
    backgroundColor: "#fff7ed",
    borderRadius: "14px",
    padding: "14px 16px",
    border: "1px solid #fed7aa",
  },
  validationText: {
    margin: "0 0 6px 0",
    color: "#9a3412",
    fontSize: "13px",
  },
  stepsCard: {
    marginTop: "18px",
    textAlign: "left",
    backgroundColor: "#f8fafc",
    borderRadius: "16px",
    padding: "16px",
    border: "1px solid #e2e8f0",
  },
  stepRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "10px",
  },
  stepDot: {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    backgroundColor: "#cbd5e1",
    flexShrink: 0,
  },
  stepDotActive: {
    backgroundColor: "#4f46e5",
  },
  stepText: {
    color: "#64748b",
    fontSize: "14px",
  },
  stepTextActive: {
    color: "#0f172a",
    fontWeight: "bold",
  },
  infoCard: {
    marginTop: "22px",
    backgroundColor: "#f8fafc",
    borderRadius: "16px",
    padding: "16px",
  },
  infoTitle: {
    margin: "0 0 6px 0",
    fontWeight: "bold",
    color: "#0f172a",
  },
  infoText: {
    margin: 0,
    color: "#64748b",
    fontSize: "14px",
  },
};

export default UploadPage;
