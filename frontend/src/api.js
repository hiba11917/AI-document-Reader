const API_BASE_URL =
  process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000";

export const AUTH_TOKEN_KEY = "ai-doc-reader-auth-token";
export const AUTH_USER_KEY = "ai-doc-reader-auth-user";

function getAuthToken() {
  return window.localStorage.getItem(AUTH_TOKEN_KEY) || "";
}

export function persistSession(token, user) {
  window.localStorage.setItem(AUTH_TOKEN_KEY, token);
  window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user));
}

export function clearSession() {
  window.localStorage.removeItem(AUTH_TOKEN_KEY);
  window.localStorage.removeItem(AUTH_USER_KEY);
}

export function getStoredUser() {
  const rawValue = window.localStorage.getItem(AUTH_USER_KEY);
  if (!rawValue) {
    return null;
  }

  try {
    return JSON.parse(rawValue);
  } catch (error) {
    return null;
  }
}

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

function normalizeNetworkError(error, fallbackMessage) {
  if (error instanceof Error) {
    if (error.name === "TypeError") {
      return new Error(
        "Could not reach the backend server. Please make sure the API is running and try again."
      );
    }
    return error;
  }

  return new Error(fallbackMessage);
}

function authHeaders(extraHeaders = {}) {
  const token = getAuthToken();
  return {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extraHeaders,
  };
}

async function authFetch(path, options = {}) {
  const { headers = {}, ...rest } = options;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...rest,
    headers: authHeaders(headers),
  });
  return parseResponse(response);
}

function parseNdjsonChunk(buffer, onEvent) {
  const lines = buffer.split("\n");
  const remainder = lines.pop() || "";

  for (const line of lines) {
    if (!line.trim()) {
      continue;
    }

    const parsed = JSON.parse(line);
    onEvent(parsed);
  }

  return remainder;
}

export async function login(identifier, password) {
  const data = await authFetch("/api/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ identifier, password }),
  });
  persistSession(data.token, data.user);
  return data;
}

export async function signup({ fullName, email, password }) {
  return authFetch("/api/auth/signup", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      full_name: fullName,
      email,
      password,
    }),
  });
}

export async function fetchCurrentUser() {
  const data = await authFetch("/api/auth/me");
  if (data.user) {
    window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user));
  }
  return data.user || null;
}

export async function fetchUsers() {
  const data = await authFetch("/api/users");
  return data.users || [];
}

export async function createUser(user) {
  const data = await authFetch("/api/users", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(user),
  });
  return data.users || [];
}

export async function updateUserById(userId, updates) {
  const data = await authFetch(`/api/users/${userId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(updates),
  });
  return data.users || [];
}

export async function deleteUserById(userId) {
  const data = await authFetch(`/api/users/${userId}`, {
    method: "DELETE",
  });
  return data.users || [];
}

export async function fetchDocuments() {
  const data = await authFetch("/api/documents");
  return data.documents || [];
}

export async function fetchCollections() {
  const data = await authFetch("/api/collections");
  return data.collections || [];
}

export async function createCollection(name, documentIds) {
  const data = await authFetch("/api/collections", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name,
      document_ids: documentIds,
    }),
  });
  return data.collections || [];
}

export async function updateCollectionDocuments(collectionId, documentIds) {
  const data = await authFetch(`/api/collections/${collectionId}/documents`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document_ids: documentIds,
    }),
  });
  return data.collections || [];
}

export async function deleteCollectionById(collectionId) {
  return authFetch(`/api/collections/${collectionId}`, {
    method: "DELETE",
  });
}

export async function fetchDocumentHistory(documentId) {
  const data = await authFetch(`/api/documents/${documentId}/history`);
  return data.messages || [];
}

export async function fetchCollectionHistory(collectionId) {
  const data = await authFetch(`/api/collections/${collectionId}/history`);
  return data.messages || [];
}

export async function clearDocumentHistory(documentId) {
  return authFetch(`/api/documents/${documentId}/history`, {
    method: "DELETE",
  });
}

export async function clearCollectionHistory(collectionId) {
  return authFetch(`/api/collections/${collectionId}/history`, {
    method: "DELETE",
  });
}

export async function deleteDocumentHistoryMessage(documentId, messageId) {
  return authFetch(`/api/documents/${documentId}/history/${messageId}`, {
    method: "DELETE",
  });
}

export async function deleteCollectionHistoryMessage(collectionId, messageId) {
  return authFetch(`/api/collections/${collectionId}/history/${messageId}`, {
    method: "DELETE",
  });
}

export async function fetchSettings() {
  const response = await fetch(`${API_BASE_URL}/api/settings`);
  return parseResponse(response);
}

export async function uploadDocument(file) {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/documents/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: formData,
    });

    const data = await parseResponse(response);
    return data.document;
  } catch (error) {
    throw normalizeNetworkError(error, "Upload failed.");
  }
}

export async function deleteDocumentById(documentId) {
  return authFetch(`/api/documents/${documentId}`, {
    method: "DELETE",
  });
}

export async function askQuestion(question, documentId, options = {}) {
  return authFetch("/api/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      document_id: documentId,
      collection_id: options.collectionId,
      top_k: options.topK,
    }),
  });
}

export async function askQuestionStream(question, documentId, handlers = {}, options = {}) {
  const response = await fetch(`${API_BASE_URL}/api/ask/stream`, {
    method: "POST",
    headers: authHeaders({
      "Content-Type": "application/json",
    }),
    body: JSON.stringify({
      question,
      document_id: documentId,
      collection_id: options.collectionId,
      top_k: options.topK,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "Request failed.");
  }

  if (!response.body) {
    throw new Error("Streaming is not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload = null;

  function processEvent(event) {
    if (event.type === "token") {
      handlers.onToken?.(event.delta);
      return;
    }

    if (event.type === "status") {
      handlers.onStatus?.(event);
      return;
    }

    if (event.type === "meta") {
      handlers.onMeta?.(event);
      return;
    }

    if (event.type === "done") {
      finalPayload = event;
      handlers.onDone?.(event);
      return;
    }

    if (event.type === "error") {
      throw new Error(event.error || "The stream failed.");
    }
  }

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      buffer = parseNdjsonChunk(buffer, processEvent);
    }
  } finally {
    reader.releaseLock();
  }

  if (buffer.trim()) {
    parseNdjsonChunk(`${buffer}\n`, processEvent);
  }

  if (!finalPayload) {
    throw new Error("The answer stream ended before the final response arrived.");
  }

  return finalPayload;
}
