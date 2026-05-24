import React from "react";
import { useMemo, useState } from "react";

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

function formatDate(value) {
  if (!value) {
    return "Unknown";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatFileType(value) {
  return (value || "file").toUpperCase();
}

function LandingPage({
  currentUser,
  users,
  documents,
  documentsLoading,
  documentsError,
  deletingDocumentId,
  collections,
  deletingCollectionId,
  onLogout,
  onGoToUpload,
  onGoToAsk,
  onGoToAskCollection,
  onDeleteDocument,
  onCreateCollection,
  onUpdateCollectionDocuments,
  onDeleteCollection,
  onCreateUser,
  onUpdateUser,
  onDeleteUser,
}) {
  const [collectionName, setCollectionName] = useState("");
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [editingCollectionId, setEditingCollectionId] = useState("");
  const [savingCollection, setSavingCollection] = useState(false);
  const [userForm, setUserForm] = useState({
    fullName: "",
    email: "",
    password: "",
    role: "user",
    isActive: true,
    authSource: "local",
  });
  const [editingUserId, setEditingUserId] = useState("");
  const [savingUser, setSavingUser] = useState(false);
  const [userError, setUserError] = useState("");
  const [showUserPassword, setShowUserPassword] = useState(false);
  const hasDocuments = documents.length > 0;
  const isAdmin = currentUser?.role === "admin";
  const storageStatus = documentsError ? "Not Active" : "Active";
  const editingCollection = useMemo(
    () => collections.find((collection) => collection.id === editingCollectionId) || null,
    [collections, editingCollectionId]
  );

  const toggleDocumentSelection = (documentId) => {
    setSelectedDocumentIds((currentIds) =>
      currentIds.includes(documentId)
        ? currentIds.filter((id) => id !== documentId)
        : [...currentIds, documentId]
    );
  };

  const handleCreateCollection = async () => {
    if (!collectionName.trim() || selectedDocumentIds.length === 0 || savingCollection) {
      return;
    }

    setSavingCollection(true);
    try {
      await onCreateCollection(collectionName, selectedDocumentIds);
      setCollectionName("");
      setSelectedDocumentIds([]);
    } finally {
      setSavingCollection(false);
    }
  };

  const handleCollectionDelete = async (collectionId) => {
    if (!window.confirm("Delete this collection?")) {
      return;
    }
    await onDeleteCollection(collectionId);
    if (editingCollectionId === collectionId) {
      setEditingCollectionId("");
      setSelectedDocumentIds([]);
    }
  };

  const handleEditCollection = (collectionId) => {
    const collection = collections.find((item) => item.id === collectionId);
    setEditingCollectionId(collectionId);
    setSelectedDocumentIds(collection?.document_ids || []);
  };

  const resetUserForm = () => {
    setEditingUserId("");
    setUserForm({
      fullName: "",
      email: "",
      password: "",
      role: "user",
      isActive: true,
      authSource: "local",
    });
    setUserError("");
    setShowUserPassword(false);
  };

  const handleEditUser = (user) => {
    if (user.id === "admin") {
      return;
    }
    setEditingUserId(user.id);
    setUserForm({
      fullName: user.full_name || "",
      email: user.email || "",
      password: "",
      role: user.role || "user",
      isActive: Boolean(user.is_active),
      authSource: user.auth_source || "local",
    });
    setUserError("");
  };

  const handleSaveUser = async () => {
    if (!userForm.fullName.trim() || !userForm.email.trim()) {
      setUserError("Full name and email are required.");
      return;
    }
    if (!editingUserId && userForm.authSource === "local" && !userForm.password.trim()) {
      setUserError("Password is required when creating a user.");
      return;
    }

    setSavingUser(true);
    setUserError("");
    try {
      const payload = {
        full_name: userForm.fullName.trim(),
        email: userForm.email.trim(),
        role: userForm.role,
        is_active: userForm.isActive,
        auth_source: userForm.authSource,
      };
      if (userForm.password.trim()) {
        payload.password = userForm.password;
      }

      if (editingUserId) {
        await onUpdateUser(editingUserId, payload);
      } else {
        await onCreateUser(payload);
      }
      resetUserForm();
    } catch (error) {
      setUserError(error.message || "Could not save the user.");
    } finally {
      setSavingUser(false);
    }
  };

  const handleDeleteUser = async (user) => {
    if (user.id === "admin") {
      return;
    }
    if (!window.confirm(`Delete ${user.full_name || user.email}?`)) {
      return;
    }
    try {
      await onDeleteUser(user.id);
      if (editingUserId === user.id) {
        resetUserForm();
      }
    } catch (error) {
      setUserError(error.message || "Could not delete the user.");
    }
  };

  const handleSaveCollectionDocuments = async () => {
    if (!editingCollectionId || selectedDocumentIds.length === 0 || savingCollection) {
      return;
    }
    setSavingCollection(true);
    try {
      await onUpdateCollectionDocuments(editingCollectionId, selectedDocumentIds);
      setEditingCollectionId("");
      setSelectedDocumentIds([]);
    } finally {
      setSavingCollection(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.heroCard}>
          <div>
            <div style={styles.badge}>AI Document Reader</div>
            <h1 style={styles.title}>Document Dashboard</h1>
            <p style={styles.subtitle}>
              Upload, manage, and ask questions about your documents.
            </p>
          </div>

          <div style={styles.topButtonGroup}>
            <button style={styles.primaryButton} onClick={onGoToUpload}>
              Upload Document
            </button>
            <button
              style={{
                ...styles.secondaryButton,
                ...(hasDocuments ? null : styles.disabledButton),
              }}
              onClick={() => onGoToAsk()}
              disabled={!hasDocuments}
            >
              Ask Question
            </button>
            <button style={styles.logoutButton} onClick={onLogout}>
              Log Out
            </button>
          </div>
        </div>

        {documentsError && <div style={styles.errorBanner}>{documentsError}</div>}

        <div style={styles.statsRow}>
          <div style={styles.statCard}>
            <p style={styles.statLabel}>Total Documents</p>
            <h3 style={styles.statValue}>{documents.length}</h3>
          </div>
          <div style={styles.statCard}>
            <p style={styles.statLabel}>Ready for Q&A</p>
            <h3 style={styles.statValue}>{documents.length}</h3>
          </div>
          <div style={styles.statCard}>
            <p style={styles.statLabel}>Storage Status</p>
            <h3 style={styles.statValue}>{storageStatus}</h3>
          </div>
        </div>

        <div style={styles.collectionCard}>
          <div style={styles.tableHeader}>
            <div>
              <h2 style={styles.sectionTitle}>Collections</h2>
              <p style={styles.sectionSubtitle}>
                Group documents together, then ask questions across the whole collection.
              </p>
            </div>
          </div>

          <div style={styles.collectionGrid}>
            <div style={styles.collectionBuilder}>
              <label style={styles.collectionLabel}>Collection Name</label>
              <input
                style={styles.collectionInput}
                type="text"
                value={collectionName}
                onChange={(event) => setCollectionName(event.target.value)}
                placeholder="Enter a name for the collection"
              />
              <p style={styles.collectionHint}>
                Choose a short name that describes the group of documents.
              </p>
              <div style={styles.collectionSelectionList}>
                {documents.length === 0 ? (
                  <p style={styles.emptyState}>Upload documents first to create a collection.</p>
                ) : (
                  documents.map((document) => (
                    <label key={document.id} style={styles.collectionCheckboxRow}>
                      <input
                        type="checkbox"
                        checked={selectedDocumentIds.includes(document.id)}
                        onChange={() => toggleDocumentSelection(document.id)}
                      />
                      <span>{document.filename}</span>
                    </label>
                  ))
                )}
              </div>
              {editingCollection ? (
                <div style={styles.actionGroup}>
                  <button
                    style={{
                      ...styles.primaryAction,
                      ...(savingCollection ? styles.disabledButton : null),
                    }}
                    onClick={handleSaveCollectionDocuments}
                    disabled={savingCollection}
                  >
                    {savingCollection ? "Saving..." : "Save Collection Documents"}
                  </button>
                  <button
                    style={styles.cancelButton}
                    onClick={() => {
                      setEditingCollectionId("");
                      setSelectedDocumentIds([]);
                    }}
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  style={{
                    ...styles.primaryAction,
                    ...((!collectionName.trim() || selectedDocumentIds.length === 0 || savingCollection)
                      ? styles.disabledButton
                      : null),
                  }}
                  onClick={handleCreateCollection}
                  disabled={!collectionName.trim() || selectedDocumentIds.length === 0 || savingCollection}
                >
                  {savingCollection ? "Creating..." : "Create Collection"}
                </button>
              )}
            </div>

            <div style={styles.collectionList}>
              {collections.length === 0 ? (
                <div style={styles.emptyState}>No collections created yet.</div>
              ) : (
                collections.map((collection) => (
                  <div key={collection.id} style={styles.collectionItem}>
                    <div>
                      <h3 style={styles.collectionName}>{collection.name}</h3>
                      <p style={styles.collectionMeta}>
                        {collection.document_count} document{collection.document_count === 1 ? "" : "s"}
                      </p>
                      <p style={styles.collectionDocs}>
                        {(collection.documents || []).map((doc) => doc.filename).join(", ") || "No documents yet"}
                      </p>
                    </div>
                    <div style={styles.collectionActionGroup}>
                      <button
                        style={styles.askButton}
                        onClick={() => onGoToAskCollection(collection.id)}
                      >
                        Ask Collection
                      </button>
                      <button
                        style={styles.secondaryOutlineButton}
                        onClick={() => handleEditCollection(collection.id)}
                      >
                        Edit Docs
                      </button>
                      <button
                        style={{
                          ...styles.deleteButton,
                          ...(deletingCollectionId === collection.id ? styles.disabledButton : null),
                        }}
                        onClick={() => handleCollectionDelete(collection.id)}
                        disabled={deletingCollectionId === collection.id}
                      >
                        {deletingCollectionId === collection.id ? "Deleting..." : "Delete"}
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {isAdmin && (
          <div style={styles.collectionCard}>
            <div style={styles.tableHeader}>
              <div>
                <h2 style={styles.sectionTitle}>Users</h2>
                <p style={styles.sectionSubtitle}>
                  Admin can create, edit, disable, and delete normal users.
                </p>
              </div>
            </div>

            <div style={styles.collectionGrid}>
              <div style={styles.collectionBuilder}>
                <label style={styles.collectionLabel}>Full Name</label>
                <input
                  style={styles.collectionInput}
                  type="text"
                  value={userForm.fullName}
                  onChange={(event) =>
                    setUserForm((current) => ({ ...current, fullName: event.target.value }))
                  }
                  placeholder="Enter the user's name"
                />

                <label style={styles.collectionLabel}>Email</label>
                <input
                  style={styles.collectionInput}
                  type="email"
                  value={userForm.email}
                  onChange={(event) =>
                    setUserForm((current) => ({ ...current, email: event.target.value }))
                  }
                  placeholder="Enter the user's email"
                />

                <label style={styles.collectionLabel}>Auth Source</label>
                <select
                  style={styles.collectionInput}
                  value={userForm.authSource}
                  onChange={(event) =>
                    setUserForm((current) => ({ ...current, authSource: event.target.value }))
                  }
                >
                  <option value="local">Local User</option>
                  <option value="active_directory">Active Directory</option>
                </select>

                <label style={styles.collectionLabel}>
                  {userForm.authSource === "active_directory"
                    ? "Password"
                    : editingUserId
                    ? "Reset Password (Optional)"
                    : "Password"}
                </label>
                <div style={styles.passwordRow}>
                  <input
                    style={styles.passwordInput}
                    type={showUserPassword ? "text" : "password"}
                    value={userForm.password}
                    onChange={(event) =>
                      setUserForm((current) => ({ ...current, password: event.target.value }))
                    }
                    placeholder={
                      userForm.authSource === "active_directory"
                        ? "Leave blank if the user signs in with AD credentials"
                        : editingUserId
                        ? "Leave blank to keep current password"
                        : "Enter a password"
                    }
                  />
                  <button
                    type="button"
                    style={styles.eyeButton}
                    onClick={() => setShowUserPassword((current) => !current)}
                    aria-label={showUserPassword ? "Hide password" : "Show password"}
                  >
                    <EyeIcon open={showUserPassword} />
                  </button>
                </div>

                <label style={styles.collectionLabel}>Role</label>
                <select
                  style={styles.collectionInput}
                  value={userForm.role}
                  onChange={(event) =>
                    setUserForm((current) => ({ ...current, role: event.target.value }))
                  }
                >
                  <option value="user">Normal User</option>
                  <option value="admin">Admin User</option>
                </select>

                <label style={styles.collectionCheckboxRow}>
                  <input
                    type="checkbox"
                    checked={userForm.isActive}
                    onChange={(event) =>
                      setUserForm((current) => ({ ...current, isActive: event.target.checked }))
                    }
                  />
                  <span>User is active</span>
                </label>

                {userError && <div style={styles.inlineError}>{userError}</div>}

                <div style={styles.actionGroup}>
                  <button
                    style={{
                      ...styles.primaryAction,
                      ...(savingUser ? styles.disabledButton : null),
                    }}
                    onClick={handleSaveUser}
                    disabled={savingUser}
                  >
                    {savingUser
                      ? editingUserId
                        ? "Saving..."
                        : "Creating..."
                      : editingUserId
                      ? "Save User"
                      : "Create User"}
                  </button>
                  {editingUserId && (
                    <button style={styles.cancelButton} onClick={resetUserForm}>
                      Cancel
                    </button>
                  )}
                </div>
              </div>

              <div style={styles.collectionList}>
                {users.map((user) => (
                  <div key={user.id} style={styles.collectionItem}>
                    <div>
                      <h3 style={styles.collectionName}>{user.full_name}</h3>
                      <p style={styles.collectionMeta}>
                        {user.email} • {user.role === "admin" ? "Admin" : "Normal User"} •{" "}
                        {user.is_active ? "Active" : "Inactive"}
                      </p>
                      <p style={styles.collectionDocs}>
                        Username: {user.username}
                        {user.auth_source ? ` • Source: ${user.auth_source === "active_directory" ? "Active Directory" : "Local"}` : ""}
                      </p>
                    </div>
                    <div style={styles.collectionActionGroup}>
                      <button
                        style={{
                          ...styles.secondaryOutlineButton,
                          ...(user.id === "admin" ? styles.disabledButton : null),
                        }}
                        onClick={() => handleEditUser(user)}
                        disabled={user.id === "admin"}
                      >
                        Edit User
                      </button>
                      <button
                        style={{
                          ...styles.deleteButton,
                          ...(user.id === "admin" ? styles.disabledButton : null),
                        }}
                        onClick={() => handleDeleteUser(user)}
                        disabled={user.id === "admin"}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        <div style={styles.tableCard}>
          <div style={styles.tableHeader}>
            <div>
              <h2 style={styles.sectionTitle}>Uploaded Documents</h2>
              <p style={styles.sectionSubtitle}>
                View, remove, or ask questions about your files.
              </p>
            </div>
          </div>

          <div style={styles.tableWrapper}>
            {documentsLoading ? (
              <div style={styles.emptyState}>Loading documents...</div>
            ) : (
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Document Name</th>
                    <th style={styles.th}>Type</th>
                    <th style={styles.th}>Uploaded On</th>
                    <th style={styles.th}>Status</th>
                    <th style={styles.th}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {documents.map((document) => (
                    <tr key={document.id} style={styles.row}>
                      <td style={styles.td}>
                        <div style={styles.fileName}>{document.filename}</div>
                      </td>
                      <td style={styles.td}>
                        <span style={styles.fileType}>
                          {formatFileType(document.file_type)}
                        </span>
                      </td>
                      <td style={styles.td}>{formatDate(document.created_at)}</td>
                      <td style={styles.td}>
                        <span style={styles.statusBadge}>Indexed</span>
                      </td>
                      <td style={styles.td}>
                        <div style={styles.actionGroup}>
                          <button
                            style={styles.askButton}
                            onClick={() => onGoToAsk(document.id)}
                          >
                            Ask Question
                          </button>
                          <button
                            style={{
                              ...styles.deleteButton,
                              ...(deletingDocumentId === document.id
                                ? styles.disabledButton
                                : null),
                            }}
                            onClick={() => onDeleteDocument(document.id)}
                            disabled={deletingDocumentId === document.id}
                          >
                            {deletingDocumentId === document.id
                              ? "Deleting..."
                              : "Delete"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            {!documentsLoading && documents.length === 0 && (
              <div style={styles.emptyState}>No documents uploaded yet.</div>
            )}
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
    padding: "32px 20px",
    fontFamily: "Arial, sans-serif",
  },
  container: {
    maxWidth: "1180px",
    margin: "0 auto",
  },
  heroCard: {
    background: "linear-gradient(135deg, #1e3a8a, #4338ca)",
    color: "#fff",
    padding: "32px",
    borderRadius: "28px",
    boxShadow: "0 20px 50px rgba(30, 58, 138, 0.20)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: "20px",
    flexWrap: "wrap",
    marginBottom: "24px",
  },
  badge: {
    display: "inline-block",
    backgroundColor: "rgba(255,255,255,0.15)",
    padding: "8px 14px",
    borderRadius: "999px",
    fontSize: "13px",
    marginBottom: "14px",
  },
  title: {
    margin: 0,
    fontSize: "38px",
  },
  subtitle: {
    marginTop: "10px",
    color: "rgba(255,255,255,0.85)",
    maxWidth: "560px",
    lineHeight: 1.6,
  },
  topButtonGroup: {
    display: "flex",
    gap: "12px",
    flexWrap: "wrap",
  },
  primaryButton: {
    padding: "13px 18px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#ffffff",
    color: "#1e3a8a",
    fontWeight: "bold",
    cursor: "pointer",
  },
  secondaryButton: {
    padding: "13px 18px",
    border: "1px solid rgba(255,255,255,0.4)",
    borderRadius: "12px",
    backgroundColor: "rgba(255,255,255,0.10)",
    color: "#ffffff",
    fontWeight: "bold",
    cursor: "pointer",
  },
  logoutButton: {
    padding: "13px 18px",
    border: "none",
    borderRadius: "12px",
    backgroundColor: "#ef4444",
    color: "#ffffff",
    fontWeight: "bold",
    cursor: "pointer",
  },
  disabledButton: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  errorBanner: {
    marginBottom: "20px",
    padding: "14px 16px",
    borderRadius: "14px",
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#b91c1c",
    fontWeight: "bold",
  },
  statsRow: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: "18px",
    marginBottom: "24px",
  },
  statCard: {
    backgroundColor: "#ffffff",
    padding: "24px",
    borderRadius: "20px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
  },
  statLabel: {
    margin: 0,
    fontSize: "14px",
    color: "#64748b",
  },
  statValue: {
    margin: "10px 0 0 0",
    fontSize: "28px",
    color: "#0f172a",
  },
  tableCard: {
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
    overflow: "hidden",
  },
  collectionCard: {
    backgroundColor: "#ffffff",
    borderRadius: "24px",
    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.08)",
    overflow: "hidden",
    marginBottom: "24px",
  },
  collectionGrid: {
    display: "grid",
    gridTemplateColumns: "minmax(260px, 340px) 1fr",
    gap: "18px",
    padding: "0 18px 18px 18px",
  },
  collectionBuilder: {
    border: "1px solid #e2e8f0",
    borderRadius: "18px",
    padding: "16px",
    backgroundColor: "#f8fafc",
  },
  collectionInput: {
    width: "100%",
    boxSizing: "border-box",
    borderRadius: "12px",
    border: "1px solid #cbd5e1",
    padding: "12px 14px",
    fontSize: "14px",
    marginBottom: "8px",
  },
  passwordRow: {
    position: "relative",
    marginBottom: "8px",
  },
  passwordInput: {
    width: "100%",
    boxSizing: "border-box",
    borderRadius: "12px",
    border: "1px solid #cbd5e1",
    padding: "12px 14px",
    paddingRight: "52px",
    fontSize: "14px",
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
  collectionLabel: {
    display: "block",
    marginBottom: "8px",
    fontWeight: "bold",
    fontSize: "14px",
    color: "#0f172a",
  },
  collectionHint: {
    margin: "0 0 12px 0",
    fontSize: "13px",
    color: "#64748b",
    lineHeight: 1.5,
  },
  inlineError: {
    marginBottom: "12px",
    padding: "10px 12px",
    borderRadius: "12px",
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#b91c1c",
    fontSize: "13px",
    fontWeight: "bold",
  },
  collectionSelectionList: {
    maxHeight: "240px",
    overflowY: "auto",
    marginBottom: "12px",
  },
  collectionCheckboxRow: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    fontSize: "14px",
    color: "#0f172a",
    padding: "8px 0",
  },
  primaryAction: {
    padding: "12px 14px",
    backgroundColor: "#1d4ed8",
    color: "#ffffff",
    border: "none",
    borderRadius: "12px",
    fontWeight: "bold",
    cursor: "pointer",
  },
  cancelButton: {
    padding: "12px 14px",
    backgroundColor: "#e2e8f0",
    color: "#0f172a",
    border: "none",
    borderRadius: "12px",
    fontWeight: "bold",
    cursor: "pointer",
  },
  collectionList: {
    display: "grid",
    gap: "14px",
  },
  collectionItem: {
    border: "1px solid #e2e8f0",
    borderRadius: "18px",
    padding: "18px 20px",
    display: "flex",
    justifyContent: "space-between",
    gap: "20px",
    alignItems: "center",
    backgroundColor: "#ffffff",
    minHeight: "120px",
  },
  collectionName: {
    margin: "0 0 6px 0",
    fontSize: "18px",
    color: "#0f172a",
  },
  collectionMeta: {
    margin: "0 0 8px 0",
    color: "#475569",
    fontSize: "13px",
  },
  collectionDocs: {
    margin: 0,
    color: "#64748b",
    fontSize: "13px",
    lineHeight: 1.5,
    maxWidth: "620px",
  },
  secondaryOutlineButton: {
    padding: "10px 14px",
    backgroundColor: "#ffffff",
    color: "#1d4ed8",
    border: "1px solid #bfdbfe",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  tableHeader: {
    padding: "24px 26px 10px 26px",
  },
  sectionTitle: {
    margin: 0,
    color: "#0f172a",
    fontSize: "26px",
  },
  sectionSubtitle: {
    marginTop: "8px",
    color: "#64748b",
    fontSize: "14px",
  },
  tableWrapper: {
    overflowX: "auto",
    padding: "0 16px 18px 16px",
  },
  table: {
    width: "100%",
    borderCollapse: "separate",
    borderSpacing: 0,
  },
  th: {
    textAlign: "left",
    padding: "16px",
    color: "#475569",
    fontSize: "14px",
    backgroundColor: "#f8fafc",
    borderBottom: "1px solid #e2e8f0",
  },
  row: {
    backgroundColor: "#ffffff",
  },
  td: {
    padding: "18px 16px",
    borderBottom: "1px solid #edf2f7",
    color: "#0f172a",
    fontSize: "14px",
    verticalAlign: "middle",
  },
  fileName: {
    fontWeight: "bold",
  },
  fileType: {
    display: "inline-block",
    padding: "6px 10px",
    backgroundColor: "#eff6ff",
    color: "#1d4ed8",
    borderRadius: "999px",
    fontSize: "12px",
    fontWeight: "bold",
  },
  statusBadge: {
    display: "inline-block",
    padding: "6px 10px",
    backgroundColor: "#ecfdf5",
    color: "#166534",
    borderRadius: "999px",
    fontSize: "12px",
    fontWeight: "bold",
  },
  actionGroup: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },
  collectionActionGroup: {
    display: "flex",
    gap: "12px",
    flexWrap: "nowrap",
    alignItems: "center",
    justifyContent: "flex-end",
    minWidth: "330px",
  },
  askButton: {
    padding: "10px 14px",
    backgroundColor: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  deleteButton: {
    padding: "10px 14px",
    backgroundColor: "#fee2e2",
    color: "#b91c1c",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  emptyState: {
    textAlign: "center",
    padding: "30px",
    color: "#64748b",
  },
};

export default LandingPage;
