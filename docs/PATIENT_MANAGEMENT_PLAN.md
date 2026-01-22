# Patient Management System Implementation Plan

## 1. Executive Summary
This document outlines the architecture and implementation plan for adding a patient management layer to the MediLens platform. This feature allows system users (doctors, testers) to manage distinct patient profiles, save assessments to specific patients, and track longitudinal history.

## 2. Architecture & Data Model

### 2.1. Concept
*   **System User (Clerk User):** The person logged into the application (e.g., the doctor or technician).
*   **Patient (Medical Entity):** The subject of the medical assessment.
*   **Relationship:** One System User can manage multiple Patients. (In the future, Patients might be shared, but for now, we assume a pool of patients accessible to the platform).

### 2.2. Database Schema Changes (Neon/Postgres)

**New Table: `patients`**
```sql
CREATE TABLE patients (
  id SERIAL PRIMARY KEY,
  full_name VARCHAR(255) NOT NULL,
  phone_number VARCHAR(50) UNIQUE NOT NULL, -- Primary Identifier
  age INTEGER,
  gender VARCHAR(20),
  medical_notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Modified Table: `assessments` & `audio_recordings`**
*   Add `patient_id` column (Foreign Key linked to `patients.id`).
*   `created_by` column (optional, to track which System User performed the test).

### 2.3. Backend API (FastAPI)

**New Router: `app/routers/patients.py`**
*   `POST /api/patients`: Create new patient.
*   `GET /api/patients`: Search patients (query by name/phone).
*   `GET /api/patients/{id}`: Get patient details.
*   `GET /api/patients/{id}/history`: Get all assessments for a patient.

**Middleware/Context**
*   Ensure that when an assessment pipeline is triggered (`/api/{pipeline}/analyze`), the `patient_id` is passed and saved.

## 3. Frontend Architecture (Next.js)

### 3.1. State Management
*   **`PatientContext`**: A global context provider to manage the "Active Patient".
    *   State: `activePatient` (null | Patient Object)
    *   Actions: `setActivePatient(patient)`, `clearActivePatient()`
    *   Persistence: Use `localStorage` to persist active patient across reloads (optional but recommended for UX).

### 3.2. UI Components

**1. Header Enhancements (`Header.tsx`)**
*   **Right Side:** Add a "Patient Control" area.
*   **State: No Patient Active:**
    *   Button: `[+ New Patient]` -> Opens creation modal.
    *   Search: Simple icon/input to find existing patient.
*   **State: Patient Active:**
    *   Display: `Active: Jane Doe (Age 32)`
    *   Actions: `[Change]` (Switch patient), `[History]` (View logs).

**2. New Patient Modal (`NewPatientModal.tsx`)**
*   Fields: Full Name, Phone Number (Required, Unique), Age, Gender (Optional).
*   Action: Saves to DB, sets as `activePatient` immediately upon success.

**3. Patient Search/Selector (`PatientSelector.tsx`)**
*   Combobox/Autocomplete input to search by name or phone.

## 4. Implementation Steps

### Phase 1: Backend Core
1.  **Schema:** Create `patients` model in `models/patient.py`.
2.  **Migration:** Update database schema (using `init_db` or migration script).
3.  **API:** Implement `routers/patients.py`.
4.  **Integration:** Update `pipelines/*` to accept and save `patient_id`.

### Phase 2: Frontend State & Components
1.  **Context:** Create `context/PatientContext.tsx`.
2.  **API Client:** Add patient service methods (create, search, get history).
3.  **Components:** Build `NewPatientModal` and `PatientSearch`.

### Phase 3: UI Integration
1.  **Header:** Integrate components into `Header.tsx`.
2.  **Pipelines:** Ensure assessment submission calls include the `activePatient.id`.

## 5. User Scenarios

**Scenario A: New Patient Assessment**
1.  Doctor logs in.
2.  Clicks `+ New Patient` in header.
3.  Enters "John Doe", "555-0101", Age 45.
4.  System creates record, sets "John Doe" as Active.
5.  Header shows "Active: John Doe".
6.  Doctor navigates to "Retinal Analysis".
7.  Performs scan. Result is saved linked to John Doe.

**Scenario B: Returning Patient**
1.  Doctor logs in.
2.  Uses Search bar in header to find "555-0101".
3.  Selects "John Doe".
4.  Header updates to active state.
5.  Doctor views "History" to see past results.
