/**
 * Drug-Pred AI — TypeScript Type Definitions
 * Shared interfaces cho toàn bộ frontend
 */

// ============================================
// Auth
// ============================================
export interface User {
  id: string;
  username: string;
  email: string;
  fullName: string;
  role: "admin" | "doctor" | "nurse" | "researcher";
  isActive: boolean;
  createdAt: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  accessToken: string;
  tokenType: string;
  user: User;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  fullName: string;
  role: "doctor" | "nurse" | "researcher";
}

// ============================================
// Patient
// ============================================
export interface Patient {
  id: string;
  patientCode: string;
  fullName: string;
  dateOfBirth: string;
  gender: "male" | "female" | "other";
  phone?: string;
  address?: string;
  bloodType?: "A+" | "A-" | "B+" | "B-" | "AB+" | "AB-" | "O+" | "O-";
  allergies?: string[];
  chronicDiseases?: string[];
  createdBy?: string;
  createdAt: string;
}

export interface CreatePatientRequest {
  fullName: string;
  dateOfBirth: string;
  gender: "male" | "female" | "other";
  phone?: string;
  address?: string;
  bloodType?: string;
  allergies?: string[];
  chronicDiseases?: string[];
}

// ============================================
// Medical Record
// ============================================
export interface VitalSigns {
  temperature?: number;
  bloodPressure?: string;
  heartRate?: number;
  respiratoryRate?: number;
  spo2?: number;
}

export interface MedicalRecord {
  id: string;
  recordCode: string;
  patientId: string;
  createdBy: string;
  chiefComplaint: string;
  description?: string;
  symptomsDuration?: string;
  vitalSigns?: VitalSigns;
  diagnosis?: string;
  diagnosisIcd?: string;
  severity: "mild" | "moderate" | "severe" | "critical";
  status: "pending" | "predicted" | "confirmed" | "archived";
  createdAt: string;
}

export interface CreateRecordRequest {
  patientId: string;
  chiefComplaint: string;
  description?: string;
  symptomsDuration?: string;
  vitalSigns?: VitalSigns;
  severity?: "mild" | "moderate" | "severe" | "critical";
}

// ============================================
// Prediction
// ============================================
export interface DrugGroupPrediction {
  drugGroupId: string;
  drugGroupName: string;
  confidence: number;
  rank: number;
}

export interface Prediction {
  id: string;
  recordId: string;
  modelConfigId: string;
  predictedGroups: DrugGroupPrediction[];
  top1GroupId?: string;
  top1Confidence?: number;
  processingTimeMs?: number;
  isConfirmed: boolean;
  confirmedGroupId?: string;
  confirmedBy?: string;
  confirmedAt?: string;
  doctorFeedback?: string;
  feedbackRating?: number;
  createdAt: string;
}

export interface ConfirmPredictionRequest {
  confirmedGroupId: string;
  doctorFeedback?: string;
  feedbackRating?: number;
}

// ============================================
// Drug Group
// ============================================
export interface DrugGroup {
  id: string;
  name: string;
  code: string;
  category: string;
  description?: string;
  commonDrugs?: string[];
  contraindications?: string[];
  sideEffects?: string[];
}

// ============================================
// Dashboard / Statistics
// ============================================
export interface DashboardStats {
  totalPatients: number;
  totalRecords: number;
  totalPredictions: number;
  confirmedPredictions: number;
  averageAccuracy: number;
  modelVersion: string;
}

// ============================================
// API Response Wrappers
// ============================================
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface ApiError {
  detail: string;
  statusCode: number;
}
