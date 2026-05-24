import React, { useEffect, useMemo, useState } from "react";
import {
  askQuestion,
  clearCollectionHistory,
  clearDocumentHistory,
  deleteCollectionHistoryMessage,
  deleteDocumentHistoryMessage,
  fetchCollectionHistory,
  fetchDocumentHistory,
  fetchSettings,
} from "./api";

function formatMessageHistory(historyMessages) {
  return historyMessages.map((message) => ({
    id: message.id,
    type: message.role === "assistant" ? "assistant" : "user",
    text: message.message,
    citation: "",
    citations: message.citations || [],
    model: message.model || "",
    createdAt: message.created_at,
    pending: false,
  }));
}

function getDisplayModelLabel(modelName) {
  if (!modelName) {
    return "";
  }
  return "AI";
}

function buildWaitingMessage(modelLabel) {
  if (!modelLabel) {
    return "Preparing your answer from the selected document...";
  }

  return `Searching the document and generating with ${modelLabel}...`;
}

function buildInlineCitationText(citations) {
  if (!citations || citations.length === 0) {
    return "";
  }

  return citations
    .slice(0, 2)
    .map((source) => {
      const page = source.page ? `Page ${source.page}` : "Page n/a";
      const preciseRef = source.source_ref || source.cell_range || page;
      return `${source.filename} (${preciseRef})`;
    })
    .join(" | ");
}

function CitationList({ citations }) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div style={styles.citationsPanel}>
      <p style={styles.citationsTitle}>Sources</p>
      {citations.slice(0, 3).map((source, index) => (
        <div key={`${source.chunk_id}-${index}`} style={styles.citationCard}>
          <div style={styles.citationHeader}>
            <span style={styles.citationFile}>{source.filename}</span>
            <span style={styles.citationMeta}>
              {source.source_ref ||
                source.cell_range ||
                (source.page ? `Page ${source.page}` : "Page n/a")}
            </span>
          </div>
          <p style={styles.citationSnippet}>{source.snippet}</p>
        </div>
      ))}
    </div>
  );
}

function AskQuestionPage({
  documents,
  collections,
  selectedDocumentId,
  selectedCollectionId,
  onSelectDocument,
  onSelectCollection,
  onBackToDashboard,
  onLogout,
}) {
  const [scope, setScope] = useState("document");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState({});
  const [messages, setMessages] = useState([]);
  const [statusMessage, setStatusMessage] = useState("");
  const [settings, setSettings] = useState({
    default_model: "",
    fallback_model: "",
  });
  const [historyLoading, setHistoryLoading] = useState(false);
  const [clearingHistory, setClearingHistory] = useState(false);
  const [deletingMessageId, setDeletingMessageId] = useState("");
  const [slowWarning, setSlowWarning] = useState("");

  const selectedDocument = useMemo(() => {
    return (
      documents.find((document) => document.id === selectedDocumentId) ||
      documents[0] ||
      null
    );
  }, [documents, selectedDocumentId]);

  const selectedCollection = useMemo(() => {
    return (
      collections.find((collection) => collection.id === selectedCollectionId) || null
    );
  }, [collections, selectedCollectionId]);

  const activeScopeLabel =
    scope === "collection" && selectedCollection
      ? selectedCollection.name
      : selectedDocument?.filename || "";

  const activeScopeMeta =
    scope === "collection" && selectedCollection
      ? `${selectedCollection.document_count} docs`
      : selectedDocument?.file_type
      ? selectedDocument.file_type.toUpperCase()
      : "FILE";

  useEffect(() => {
    let active = true;

    async function loadSettings() {
      try {
        const nextSettings = await fetchSettings();
        if (!active) {
          return;
        }
        setSettings(nextSettings);
      } catch (error) {
        if (active) {
          setSettings({ default_model: "", fallback_model: "" });
        }
      }
    }

    loadSettings();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!loading) {
      setSlowWarning("");
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setSlowWarning("It will take 1 to 3 minutes, please wait.......");
    }, 15000);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [loading]);

  useEffect(() => {
    if (!selectedDocument && documents.length > 0) {
      onSelectDocument(documents[0].id);
    }
  }, [documents, onSelectDocument, selectedDocument]);

  useEffect(() => {
    let active = true;

    async function loadHistory() {
      setStatusMessage("");

      const activeTargetId =
        scope === "collection" ? selectedCollection?.id : selectedDocument?.id;

      if (!activeTargetId) {
        setMessages([
          {
            type: "assistant",
            text:
              scope === "collection"
                ? "Create a collection first, then come back here to ask questions across multiple documents."
                : "Upload a document first, then come back here to ask questions.",
            citation: "",
            citations: [],
            pending: false,
          },
        ]);
        return;
      }

      setHistoryLoading(true);
      try {
        const historyMessages =
          scope === "collection"
            ? await fetchCollectionHistory(activeTargetId)
            : await fetchDocumentHistory(activeTargetId);
        if (!active) {
          return;
        }

        setHistory((previousHistory) => ({
          ...previousHistory,
          [`${scope}:${activeTargetId}`]: historyMessages
            .filter((message) => message.role === "user")
            .map((message) => message.message)
            .reverse(),
        }));

        if (historyMessages.length === 0) {
          setMessages([
            {
              type: "assistant",
              text:
                scope === "collection" && selectedCollection
                  ? `You are now asking questions across the "${selectedCollection.name}" collection.`
                  : `You are now asking questions about "${selectedDocument.filename}".`,
              citation: "",
              citations: [],
              pending: false,
            },
          ]);
        } else {
          setMessages(formatMessageHistory(historyMessages));
        }
      } catch (error) {
        if (!active) {
          return;
        }

        setMessages([
          {
            type: "assistant",
            text:
              error.message ||
              `Could not load this ${scope === "collection" ? "collection" : "document"} history.`,
            citation: "",
            citations: [],
            pending: false,
          },
        ]);
      } finally {
        if (active) {
          setHistoryLoading(false);
        }
      }
    }

    loadHistory();
    return () => {
      active = false;
    };
  }, [scope, selectedDocument, selectedCollection]);

  const handleDocumentChange = (event) => {
    onSelectDocument(event.target.value);
  };

  const handleCollectionChange = (event) => {
    onSelectCollection(event.target.value);
  };

  const handleAskQuestion = async () => {
    const trimmedQuestion = question.trim();
    const activeTargetId =
      scope === "collection" ? selectedCollection?.id : selectedDocument?.id;

    if (!trimmedQuestion || !activeTargetId || loading) {
      return;
    }

    setMessages((previousMessages) => [
      ...previousMessages,
      {
        type: "user",
        text: trimmedQuestion,
        citation: "",
        citations: [],
        pending: false,
      },
      {
        type: "assistant",
        text: "",
        citation: "",
        citations: [],
        model: settings.default_model || "",
        pending: true,
      },
    ]);

    setHistory((previousHistory) => ({
      ...previousHistory,
      [`${scope}:${activeTargetId}`]: [
        trimmedQuestion,
        ...(previousHistory[`${scope}:${activeTargetId}`] || []),
      ],
    }));

    setQuestion("");
    setLoading(true);
    setSlowWarning("");
    setStatusMessage(buildWaitingMessage(settings.default_model));

    try {
      const result = await askQuestion(
        trimmedQuestion,
        scope === "document" ? activeTargetId : null,
        {
          collectionId: scope === "collection" ? activeTargetId : undefined,
          topK: 1,
        }
      );

      setMessages((previousMessages) => {
        const nextMessages = [...previousMessages];
        const lastIndex = nextMessages.length - 1;
        nextMessages[lastIndex] = {
          ...nextMessages[lastIndex],
          text:
            result.answer ||
            nextMessages[lastIndex].text ||
            "I do not know based on the uploaded documents.",
          citation: buildInlineCitationText(result.sources),
          citations: result.sources || [],
          model: result.model || nextMessages[lastIndex].model,
          pending: false,
        };
        return nextMessages;
      });

      if (result.fallback_used) {
        setStatusMessage(
          `Llama was too slow, so the app used ${result.model} for this answer.`
        );
      } else {
        setStatusMessage("");
      }
    } catch (error) {
      setMessages((previousMessages) => {
        const nextMessages = [...previousMessages];
        const lastIndex = nextMessages.length - 1;
        nextMessages[lastIndex] = {
          ...nextMessages[lastIndex],
          text: error.message || "The question could not be answered.",
          citation: "",
          citations: [],
          pending: false,
        };
        return nextMessages;
      });
      setStatusMessage("");
    } finally {
      setLoading(false);
    }
  };

  const handleHistoryClick = (pastQuestion) => {
    setQuestion(pastQuestion);
  };

  const handleClearHistory = async () => {
    const activeTargetId =
      scope === "collection" ? selectedCollection?.id : selectedDocument?.id;
    if (!activeTargetId || clearingHistory) {
      return;
    }

    const confirmed = window.confirm(
      `Delete saved chat history for "${activeScopeLabel}"?`
    );
    if (!confirmed) {
      return;
    }

    setClearingHistory(true);
    try {
      if (scope === "collection") {
        await clearCollectionHistory(activeTargetId);
      } else {
        await clearDocumentHistory(activeTargetId);
      }
      setHistory((previousHistory) => ({
        ...previousHistory,
        [`${scope}:${activeTargetId}`]: [],
      }));
      setMessages([
        {
          type: "assistant",
          text:
            scope === "collection" && selectedCollection
              ? `You are now asking questions across the "${selectedCollection.name}" collection.`
              : `You are now asking questions about "${selectedDocument.filename}".`,
          citation: "",
          citations: [],
          pending: false,
        },
      ]);
      setStatusMessage("History cleared.");
    } catch (error) {
      setStatusMessage(error.message || "Could not clear the history.");
    } finally {
      setClearingHistory(false);
    }
  };

  const handleDeleteMessage = async (messageId) => {
    const activeTargetId =
      scope === "collection" ? selectedCollection?.id : selectedDocument?.id;
    if (!activeTargetId || !messageId || deletingMessageId) {
      return;
    }

    setDeletingMessageId(messageId);
    try {
      if (scope === "collection") {
        await deleteCollectionHistoryMessage(activeTargetId, messageId);
      } else {
        await deleteDocumentHistoryMessage(activeTargetId, messageId);
      }
      setMessages((previousMessages) => {
        const nextMessages = previousMessages.filter(
          (message) => message.id !== messageId
        );
        setHistory((previousHistory) => ({
          ...previousHistory,
          [`${scope}:${activeTargetId}`]: nextMessages
            .filter((message) => message.type === "user")
            .map((message) => message.text)
            .reverse(),
        }));
        return nextMessages;
      });
      setStatusMessage("Chat entry deleted.");
    } catch (error) {
      setStatusMessage(error.message || "Could not delete the chat entry.");
    } finally {
      setDeletingMessageId("");
    }
  };

  if (!selectedDocument && !selectedCollection) {
    return (
      <div style={styles.page}>
        <div style={styles.emptyCard}>
          <h1 style={styles.chatTitle}>Ask Questions</h1>
          <p style={styles.emptyText}>
            There are no documents yet. Upload one first so Ollama has
            something to search.
          </p>
          <button style={styles.backButton} onClick={onBackToDashboard}>
            Back to Dashboard
          </button>

          <button style={styles.logoutButton} onClick={onLogout}>
            Log Out
          </button>
        </div>
      </div>
    );
  }

  const activeTargetId =
    scope === "collection" ? selectedCollection?.id : selectedDocument?.id;
  const documentHistory = activeTargetId
    ? history[`${scope}:${activeTargetId}`] || []
    : [];

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <h2 style={styles.sidebarTitle}>History</h2>
            <p style={styles.sidebarSubtitle}>Questions for this document</p>
          </div>

          <div style={styles.documentBox}>
            <label style={styles.label}>Ask Scope</label>
            <select
              style={{ ...styles.select, marginBottom: "12px" }}
              value={scope}
              onChange={(event) => setScope(event.target.value)}
            >
              <option value="document">Single Document</option>
              <option value="collection" disabled={collections.length === 0}>
                Collection
              </option>
            </select>

            <label style={styles.label}>Selected Document</label>
            {scope === "collection" ? (
              <select
                style={styles.select}
                value={selectedCollection?.id || ""}
                onChange={handleCollectionChange}
              >
                {collections.map((collection) => (
                  <option key={collection.id} value={collection.id}>
                    {collection.name}
                  </option>
                ))}
              </select>
            ) : (
              <select
                style={styles.select}
                value={selectedDocument?.id || ""}
                onChange={handleDocumentChange}
              >
                {documents.map((document) => (
                  <option key={document.id} value={document.id}>
                    {document.filename}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div style={styles.historyList}>
            {historyLoading ? (
              <p style={styles.emptyText}>Loading saved questions...</p>
            ) : documentHistory.length === 0 ? (
              <p style={styles.emptyText}>No previous questions yet.</p>
            ) : (
              documentHistory.map((item, index) => (
                <button
                  key={`${selectedDocument.id}-${index}`}
                  style={styles.historyItem}
                  onClick={() => handleHistoryClick(item)}
                >
                  {item}
                </button>
              ))
            )}
          </div>

          <button
            style={{
              ...styles.clearHistoryButton,
              ...((documentHistory.length === 0 || clearingHistory)
                ? styles.disabledButton
                : null),
            }}
            onClick={handleClearHistory}
            disabled={documentHistory.length === 0 || clearingHistory}
          >
            {clearingHistory ? "Clearing..." : "Clear History"}
          </button>

          <button style={styles.backButton} onClick={onBackToDashboard}>
            Back to Dashboard
          </button>

          <button style={styles.logoutButton} onClick={onLogout}>
            Log Out
          </button>
        </div>

        <div style={styles.chatArea}>
          <div style={styles.chatHeader}>
            <h1 style={styles.chatTitle}>Ask Questions</h1>
            <p style={styles.chatSubtitle}>
              {scope === "collection" && selectedCollection ? (
                <>
                  Ask questions across the <strong>{selectedCollection.name}</strong> collection
                </>
              ) : (
                <>
                  Ask questions about <strong>{selectedDocument?.filename || "your document"}</strong>
                </>
              )}
            </p>
            <div style={styles.activeDocBanner}>
              <span style={styles.activeDocLabel}>
                {scope === "collection" ? "Active collection" : "Active document"}
              </span>
              <span style={styles.activeDocName}>{activeScopeLabel}</span>
              <span style={styles.activeDocMeta}>{activeScopeMeta}</span>
            </div>
          </div>

          <div style={styles.messagesBox}>
            {messages.map((message, index) => (
              <div key={message.id || index} style={styles.messageBlock}>
                <div
                  style={{
                    ...styles.messageRow,
                    justifyContent:
                      message.type === "user" ? "flex-end" : "flex-start",
                  }}
                >
                  <div style={styles.messageCard}>
                    <div
                      style={{
                        ...styles.messageBubble,
                        backgroundColor:
                          message.type === "user" ? "#2563eb" : "#ffffff",
                        color: message.type === "user" ? "#ffffff" : "#0f172a",
                        border:
                          message.type === "user"
                            ? "none"
                            : "1px solid #e2e8f0",
                      }}
                    >
                      {message.model && message.type === "assistant" && (
                        <p style={styles.modelTag}>{getDisplayModelLabel(message.model)}</p>
                      )}
                      <p style={styles.messageText}>
                        {message.text || (message.pending ? "Starting answer..." : "")}
                      </p>
                      {message.citation && (
                        <p style={styles.citationText}>{message.citation}</p>
                      )}
                    </div>
                    {message.id && (
                      <button
                        style={{
                          ...styles.deleteMessageButton,
                          ...(deletingMessageId === message.id ? styles.disabledButton : null),
                        }}
                        onClick={() => handleDeleteMessage(message.id)}
                        disabled={deletingMessageId === message.id}
                      >
                        {deletingMessageId === message.id ? "Deleting..." : "Delete"}
                      </button>
                    )}
                  </div>
                </div>
                {message.type === "assistant" && (
                  <CitationList citations={message.citations} />
                )}
              </div>
            ))}

            {loading && statusMessage && (
              <div style={styles.messageRow}>
                <div style={styles.loadingBubble}>{statusMessage}</div>
              </div>
            )}

            {loading && slowWarning && (
              <div style={styles.messageRow}>
                <div style={styles.warningBubble}>{slowWarning}</div>
              </div>
            )}
          </div>

          <div style={styles.inputArea}>
            <textarea
              style={styles.textarea}
              rows="3"
              placeholder="Type your question here..."
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
            />

            <button
              style={{
                ...styles.askButton,
                ...(loading ? styles.disabledButton : null),
              }}
              onClick={handleAskQuestion}
              disabled={loading}
            >
              {loading ? "Asking..." : "Ask"}
            </button>
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
    padding: "20px",
    fontFamily: "Arial, sans-serif",
  },
  container: {
    display: "flex",
    gap: "20px",
    maxWidth: "1400px",
    margin: "0 auto",
    minHeight: "90vh",
  },
  sidebar: {
    width: "300px",
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    padding: "20px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
    display: "flex",
    flexDirection: "column",
  },
  sidebarHeader: {
    marginBottom: "18px",
  },
  sidebarTitle: {
    margin: 0,
    fontSize: "24px",
    color: "#0f172a",
  },
  sidebarSubtitle: {
    marginTop: "6px",
    color: "#64748b",
    fontSize: "14px",
  },
  documentBox: {
    marginBottom: "18px",
  },
  label: {
    display: "block",
    marginBottom: "8px",
    fontWeight: "bold",
    fontSize: "14px",
    color: "#1e293b",
  },
  select: {
    width: "100%",
    padding: "12px",
    borderRadius: "12px",
    border: "1px solid #dbeafe",
    backgroundColor: "#f8fafc",
    fontSize: "14px",
  },
  historyList: {
    flex: 1,
    overflowY: "auto",
    marginBottom: "16px",
  },
  historyItem: {
    width: "100%",
    textAlign: "left",
    padding: "12px 14px",
    marginBottom: "10px",
    borderRadius: "12px",
    border: "1px solid #e2e8f0",
    backgroundColor: "#f8fafc",
    cursor: "pointer",
    color: "#0f172a",
    fontSize: "14px",
  },
  emptyCard: {
    maxWidth: "680px",
    margin: "60px auto 0 auto",
    padding: "36px",
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
    textAlign: "center",
  },
  emptyText: {
    color: "#64748b",
    fontSize: "14px",
    lineHeight: 1.6,
  },
  backButton: {
    padding: "12px 16px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#e2e8f0",
    color: "#0f172a",
    fontWeight: "bold",
    cursor: "pointer",
  },
  logoutButton: {
    padding: "12px 16px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#ef4444",
    color: "#ffffff",
    fontWeight: "bold",
    cursor: "pointer",
    marginTop: "12px",
  },
  clearHistoryButton: {
    padding: "12px 16px",
    border: "1px solid #fecaca",
    borderRadius: "12px",
    backgroundColor: "#fff1f2",
    color: "#b91c1c",
    fontWeight: "bold",
    cursor: "pointer",
    marginBottom: "12px",
  },
  chatArea: {
    flex: 1,
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  },
  chatHeader: {
    padding: "24px 24px 16px 24px",
    borderBottom: "1px solid #e2e8f0",
  },
  chatTitle: {
    margin: 0,
    fontSize: "30px",
    color: "#0f172a",
  },
  chatSubtitle: {
    marginTop: "8px",
    color: "#64748b",
    fontSize: "14px",
  },
  activeDocBanner: {
    marginTop: "14px",
    display: "inline-flex",
    alignItems: "center",
    gap: "10px",
    padding: "10px 14px",
    borderRadius: "999px",
    backgroundColor: "#eef2ff",
    border: "1px solid #c7d2fe",
    flexWrap: "wrap",
  },
  activeDocLabel: {
    fontSize: "12px",
    fontWeight: "bold",
    color: "#4338ca",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  activeDocName: {
    fontSize: "14px",
    fontWeight: "bold",
    color: "#0f172a",
  },
  activeDocMeta: {
    fontSize: "12px",
    color: "#475569",
    backgroundColor: "#ffffff",
    borderRadius: "999px",
    padding: "4px 8px",
  },
  messagesBox: {
    flex: 1,
    padding: "24px",
    overflowY: "auto",
    backgroundColor: "#f8fafc",
  },
  messageBlock: {
    marginBottom: "16px",
  },
  messageRow: {
    display: "flex",
    marginBottom: "10px",
  },
  messageCard: {
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: "8px",
    maxWidth: "75%",
  },
  messageBubble: {
    padding: "14px 16px",
    borderRadius: "16px",
    boxShadow: "0 2px 8px rgba(15, 23, 42, 0.04)",
    whiteSpace: "pre-wrap",
  },
  deleteMessageButton: {
    border: "1px solid #fecaca",
    backgroundColor: "#fff1f2",
    color: "#b91c1c",
    borderRadius: "10px",
    padding: "8px 10px",
    fontSize: "12px",
    fontWeight: "bold",
    cursor: "pointer",
  },
  modelTag: {
    margin: "0 0 8px 0",
    fontSize: "12px",
    color: "#4f46e5",
    fontWeight: "bold",
    textTransform: "uppercase",
    letterSpacing: "0.04em",
  },
  messageText: {
    margin: 0,
    lineHeight: 1.6,
    fontSize: "15px",
  },
  citationText: {
    marginTop: "10px",
    marginBottom: 0,
    fontSize: "13px",
    color: "#64748b",
    fontStyle: "italic",
  },
  citationsPanel: {
    marginLeft: "0",
    paddingLeft: "8px",
  },
  citationsTitle: {
    margin: "0 0 8px 0",
    fontSize: "12px",
    color: "#475569",
    fontWeight: "bold",
    textTransform: "uppercase",
  },
  citationCard: {
    backgroundColor: "#ffffff",
    border: "1px solid #e2e8f0",
    borderRadius: "14px",
    padding: "12px",
    marginBottom: "8px",
    boxShadow: "0 2px 8px rgba(15, 23, 42, 0.03)",
  },
  citationHeader: {
    display: "flex",
    justifyContent: "space-between",
    gap: "12px",
    marginBottom: "8px",
  },
  citationFile: {
    fontWeight: "bold",
    color: "#0f172a",
    fontSize: "13px",
  },
  citationMeta: {
    color: "#64748b",
    fontSize: "12px",
  },
  citationSnippet: {
    margin: 0,
    color: "#334155",
    fontSize: "13px",
    lineHeight: 1.5,
  },
  loadingBubble: {
    backgroundColor: "#ffffff",
    border: "1px solid #e2e8f0",
    padding: "14px 16px",
    borderRadius: "16px",
    color: "#2563eb",
    fontWeight: "bold",
  },
  warningBubble: {
    backgroundColor: "#fff7ed",
    border: "1px solid #fdba74",
    padding: "14px 16px",
    borderRadius: "16px",
    color: "#c2410c",
    fontWeight: "bold",
    maxWidth: "80%",
  },
  inputArea: {
    padding: "18px",
    borderTop: "1px solid #e2e8f0",
    display: "flex",
    gap: "12px",
    alignItems: "flex-end",
    backgroundColor: "#ffffff",
  },
  textarea: {
    flex: 1,
    padding: "14px 16px",
    borderRadius: "14px",
    border: "1px solid #dbeafe",
    fontSize: "15px",
    backgroundColor: "#f8fafc",
    resize: "none",
    outline: "none",
  },
  askButton: {
    padding: "14px 20px",
    border: "none",
    borderRadius: "12px",
    background: "linear-gradient(135deg, #2563eb, #4f46e5)",
    color: "#ffffff",
    fontWeight: "bold",
    cursor: "pointer",
    minWidth: "100px",
  },
  disabledButton: {
    opacity: 0.7,
    cursor: "not-allowed",
  },
};

export default AskQuestionPage;
