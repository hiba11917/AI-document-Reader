import React, { useCallback, useEffect, useState } from "react";
import "./App.css";
import LoginPage from "./LoginPage";
import SignupPage from "./SignupPage";
import LandingPage from "./LandingPage";
import UploadPage from "./UploadPage";
import AskQuestionPage from "./AskQuestionPage";
import {
  AUTH_TOKEN_KEY,
  AUTH_USER_KEY,
  clearSession,
  createCollection,
  createUser,
  deleteCollectionById,
  deleteDocumentById,
  deleteUserById,
  fetchCollections,
  fetchCurrentUser,
  fetchDocuments,
  fetchUsers,
  getStoredUser,
  persistSession,
  updateCollectionDocuments,
  updateUserById,
} from "./api";

const PAGE_KEY = "ai-doc-reader-page";
const DOCUMENT_KEY = "ai-doc-reader-selected-document";
const COLLECTION_KEY = "ai-doc-reader-selected-collection";

function hasStoredSession() {
  return Boolean(window.localStorage.getItem(AUTH_TOKEN_KEY));
}

function getInitialPage() {
  const savedPage = window.localStorage.getItem(PAGE_KEY);
  if (!hasStoredSession()) {
    return "login";
  }

  if (savedPage === "landing" || savedPage === "upload" || savedPage === "ask") {
    return savedPage;
  }

  return "landing";
}

function App() {
  const [page, setPage] = useState(getInitialPage);
  const [currentUser, setCurrentUser] = useState(() => getStoredUser());
  const [users, setUsers] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [collections, setCollections] = useState([]);
  const [documentsLoading, setDocumentsLoading] = useState(false);
  const [documentsError, setDocumentsError] = useState("");
  const [selectedDocumentId, setSelectedDocumentId] = useState(() => {
    return window.localStorage.getItem(DOCUMENT_KEY);
  });
  const [selectedCollectionId, setSelectedCollectionId] = useState(() => {
    return window.localStorage.getItem(COLLECTION_KEY);
  });
  const [deletingDocumentId, setDeletingDocumentId] = useState(null);
  const [deletingCollectionId, setDeletingCollectionId] = useState(null);

  const loadUsers = useCallback(async (user) => {
    if (!user || user.role !== "admin") {
      setUsers([]);
      return;
    }

    // Only admins need user-management data, so normal users skip this extra API call entirely.
    const nextUsers = await fetchUsers();
    setUsers(nextUsers);
  }, []);

  const loadDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError("");

    try {
      const nextDocuments = await fetchDocuments();
      const nextCollections = await fetchCollections();
      setDocuments(nextDocuments);
      setCollections(nextCollections);
      setSelectedDocumentId((currentId) => {
        if (currentId && nextDocuments.some((doc) => doc.id === currentId)) {
          return currentId;
        }
        return nextDocuments[0]?.id ?? null;
      });
      setSelectedCollectionId((currentId) => {
        if (currentId && nextCollections.some((collection) => collection.id === currentId)) {
          return currentId;
        }
        return nextCollections[0]?.id ?? null;
      });
    } catch (error) {
      if ((error.message || "").toLowerCase().includes("session")) {
        clearSession();
        setCurrentUser(null);
        setUsers([]);
        setDocuments([]);
        setCollections([]);
        setSelectedDocumentId(null);
        setSelectedCollectionId(null);
        setPage("login");
        return;
      }
      setDocumentsError(error.message || "Could not load documents.");
    } finally {
      setDocumentsLoading(false);
    }
  }, []);

  useEffect(() => {
    let active = true;

    async function bootstrap() {
      // Rehydrate session, documents, and collections on refresh so the app can reopen where the user left off.
      if (!hasStoredSession()) {
        clearSession();
        if (active) {
          setCurrentUser(null);
          setPage("login");
        }
        return;
      }

      try {
        const user = await fetchCurrentUser();
        if (!active) {
          return;
        }
        setCurrentUser(user);
        await loadDocuments();
        await loadUsers(user);
      } catch (error) {
        clearSession();
        if (active) {
          setCurrentUser(null);
          setPage("login");
          setUsers([]);
        }
      }
    }

    bootstrap();
    return () => {
      active = false;
    };
  }, [loadDocuments, loadUsers]);

  useEffect(() => {
    if (page === "landing" || page === "ask" || page === "upload") {
      window.localStorage.setItem(PAGE_KEY, page);
      return;
    }

    window.localStorage.removeItem(PAGE_KEY);
  }, [page]);

  useEffect(() => {
    if (currentUser) {
      window.localStorage.setItem(AUTH_USER_KEY, JSON.stringify(currentUser));
    }
  }, [currentUser]);

  useEffect(() => {
    if (selectedDocumentId) {
      window.localStorage.setItem(DOCUMENT_KEY, selectedDocumentId);
      return;
    }

    window.localStorage.removeItem(DOCUMENT_KEY);
  }, [selectedDocumentId]);

  useEffect(() => {
    if (selectedCollectionId) {
      window.localStorage.setItem(COLLECTION_KEY, selectedCollectionId);
      return;
    }

    window.localStorage.removeItem(COLLECTION_KEY);
  }, [selectedCollectionId]);

  const handleLoginSuccess = async ({ token, user }) => {
    persistSession(token, user);
    setCurrentUser(user);
    setPage("landing");
    await loadDocuments();
    await loadUsers(user);
  };

  const handleLogout = useCallback(() => {
    clearSession();
    window.localStorage.removeItem(PAGE_KEY);
    window.localStorage.removeItem(DOCUMENT_KEY);
    window.localStorage.removeItem(COLLECTION_KEY);
    setPage("login");
    setCurrentUser(null);
    setUsers([]);
    setDocuments([]);
    setCollections([]);
    setDocumentsError("");
    setSelectedDocumentId(null);
    setSelectedCollectionId(null);
  }, []);

  const handleUploadSuccess = async (uploadedDocument) => {
    setSelectedDocumentId(uploadedDocument.id);
    setPage("landing");
    await loadDocuments();
  };

  const handleDeleteDocument = async (documentId) => {
    setDeletingDocumentId(documentId);
    setDocumentsError("");

    try {
      await deleteDocumentById(documentId);
      await loadDocuments();
    } catch (error) {
      setDocumentsError(error.message || "Could not delete the document.");
    } finally {
      setDeletingDocumentId(null);
    }
  };

  const handleGoToAsk = (documentId) => {
    if (documentId) {
      setSelectedDocumentId(documentId);
    } else if (!selectedDocumentId && documents.length > 0) {
      setSelectedDocumentId(documents[0].id);
    }
    setPage("ask");
  };

  const handleGoToAskCollection = (collectionId) => {
    if (collectionId) {
      setSelectedCollectionId(collectionId);
    } else if (!selectedCollectionId && collections.length > 0) {
      setSelectedCollectionId(collections[0].id);
    }
    setPage("ask");
  };

  const handleCreateCollection = async (name, documentIds) => {
    const nextCollections = await createCollection(name, documentIds);
    setCollections(nextCollections);
    if (nextCollections.length > 0) {
      setSelectedCollectionId(nextCollections[0].id);
    }
  };

  const handleUpdateCollectionDocuments = async (collectionId, documentIds) => {
    const nextCollections = await updateCollectionDocuments(collectionId, documentIds);
    setCollections(nextCollections);
  };

  const handleDeleteCollection = async (collectionId) => {
    setDeletingCollectionId(collectionId);
    setDocumentsError("");

    try {
      const payload = await deleteCollectionById(collectionId);
      if (payload.collections) {
        setCollections(payload.collections);
      } else {
        setCollections((currentCollections) =>
          currentCollections.filter((collection) => collection.id !== collectionId)
        );
      }
      setSelectedCollectionId((currentId) => (currentId === collectionId ? null : currentId));
    } catch (error) {
      setDocumentsError(error.message || "Could not delete the collection.");
    } finally {
      setDeletingCollectionId(null);
    }
  };

  const handleCreateUser = async (payload) => {
    const nextUsers = await createUser(payload);
    setUsers(nextUsers);
  };

  const handleUpdateUser = async (userId, payload) => {
    const nextUsers = await updateUserById(userId, payload);
    setUsers(nextUsers);
  };

  const handleDeleteUser = async (userId) => {
    const nextUsers = await deleteUserById(userId);
    setUsers(nextUsers);
  };

  if (page === "login") {
    return (
      <LoginPage
        onGoToSignup={() => setPage("signup")}
        onLoginSuccess={handleLoginSuccess}
      />
    );
  }

  if (page === "signup") {
    return <SignupPage onGoToLogin={() => setPage("login")} />;
  }

  if (page === "upload") {
    return (
      <UploadPage
        onBackToDashboard={() => setPage("landing")}
        onUploadSuccess={handleUploadSuccess}
      />
    );
  }

  if (page === "ask") {
    return (
      <AskQuestionPage
        documents={documents}
        collections={collections}
        selectedDocumentId={selectedDocumentId}
        selectedCollectionId={selectedCollectionId}
        onSelectDocument={setSelectedDocumentId}
        onSelectCollection={setSelectedCollectionId}
        onBackToDashboard={() => setPage("landing")}
        onLogout={handleLogout}
      />
    );
  }

  return (
    <LandingPage
      currentUser={currentUser}
      users={users}
      documents={documents}
      documentsLoading={documentsLoading}
      documentsError={documentsError}
      deletingDocumentId={deletingDocumentId}
      collections={collections}
      deletingCollectionId={deletingCollectionId}
      onLogout={handleLogout}
      onGoToUpload={() => setPage("upload")}
      onGoToAsk={handleGoToAsk}
      onGoToAskCollection={handleGoToAskCollection}
      onDeleteDocument={handleDeleteDocument}
      onCreateCollection={handleCreateCollection}
      onUpdateCollectionDocuments={handleUpdateCollectionDocuments}
      onDeleteCollection={handleDeleteCollection}
      onCreateUser={handleCreateUser}
      onUpdateUser={handleUpdateUser}
      onDeleteUser={handleDeleteUser}
    />
  );
}

export default App;
