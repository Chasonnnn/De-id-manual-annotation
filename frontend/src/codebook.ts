import { ENTITY_TYPES } from "./hosted/types";

export interface CodebookEntry {
  label: (typeof ENTITY_TYPES)[number];
  definition: string;
  examples: readonly [string, string];
}

export const CODEBOOK: readonly CodebookEntry[] = [
  {
    label: "NAME",
    definition: "A person's name, including first names, surnames, initials, and titles used with a name.",
    examples: ["Jordan Lee", "Dr. Maya Rivera"],
  },
  {
    label: "ADDRESS",
    definition: "A street, mailing, or residential address, including apartment and postal-code details.",
    examples: ["742 Cedar Avenue, Ithaca, NY 14850", "Apartment 3B, 19 Lake Road"],
  },
  {
    label: "DATE",
    definition: "A calendar date connected to a person or session; annotate the full date expression.",
    examples: ["October 3, 2026", "03/14/2026"],
  },
  {
    label: "PHONE_NUMBER",
    definition: "A telephone or mobile number, including an extension when it is part of the number.",
    examples: ["202-555-0142", "+1 212 555 0176 ext. 4"],
  },
  {
    label: "FAX_NUMBER",
    definition: "A number explicitly identified as a fax line.",
    examples: ["Fax: 212-555-0188", "+1 607 555 0139 (fax)"],
  },
  {
    label: "EMAIL",
    definition: "An email address associated with a person or account.",
    examples: ["jordan.lee@example.com", "tutor42@example.org"],
  },
  {
    label: "SSN",
    definition: "A U.S. Social Security number, including partial or masked forms when identified as an SSN.",
    examples: ["000-00-0000", "SSN ending in 1234"],
  },
  {
    label: "ACCOUNT_NUMBER",
    definition: "A financial, insurance, or service account number tied to a person.",
    examples: ["account 00001234", "health-plan account DEMO-HP-42"],
  },
  {
    label: "DEVICE_IDENTIFIER",
    definition: "A unique device identifier or serial number, such as an IMEI, hardware serial, or MAC address.",
    examples: ["IMEI 000000000000000", "MAC 02:00:00:00:00:01"],
  },
  {
    label: "URL",
    definition: "A web address that could identify or link to a person, account, or private resource.",
    examples: ["https://example.org/profile/demo", "portal.example.edu/u/1042"],
  },
  {
    label: "IP_ADDRESS",
    definition: "An IPv4 or IPv6 address associated with a person, device, or session.",
    examples: ["192.0.2.44", "2001:db8::42"],
  },
  {
    label: "BIOMETRIC_IDENTIFIER",
    definition: "A biometric identifier or template, including a fingerprint, voiceprint, iris, or face encoding.",
    examples: ["voiceprint VP-DEMO-204", "fingerprint template FP-DEMO-17"],
  },
  {
    label: "IMAGE",
    definition: "A full-face photograph or another image that can identify a person.",
    examples: ["[uploaded selfie]", "profile_photo_demo.jpg"],
  },
  {
    label: "IDENTIFYING_NUMBER",
    definition: "Another unique personal identifier, such as a medical-record, license, passport, or vehicle number.",
    examples: ["MRN 0000001", "driver's license X0000000"],
  },
  {
    label: "AGE",
    definition: "A person's explicit age or age expression.",
    examples: ["16 years old", "age 91"],
  },
  {
    label: "SCHOOL",
    definition: "The name of a school, college, district, or other educational institution tied to a participant.",
    examples: ["Riverview High School", "Pinecrest Community College"],
  },
  {
    label: "TUTOR_PROVIDER",
    definition: "The name of the tutoring platform, provider, or service involved in the session.",
    examples: ["BrightPath Tutoring", "StudyBridge"],
  },
  {
    label: "CUSTOMIZED_FIELD",
    definition: "A project-defined sensitive field that policy explicitly requires annotating and that has no better label.",
    examples: ["family_alias=SUNFLOWER", "cohort_secret=MAPLE-7"],
  },
  {
    label: "OTHER_LOCATIONS_IDENTIFIED",
    definition: "A city, town, neighborhood, region, country, or landmark tied to a participant but not given as an address.",
    examples: ["North Harbor", "the Lakeside neighborhood"],
  },
] as const;
