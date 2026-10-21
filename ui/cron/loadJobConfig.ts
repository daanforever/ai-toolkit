import fs from 'fs';
import path from 'path';
import YAML, { isMap, isPair, isScalar, isSeq } from 'yaml';
import type { CollectionTag, Document, ParseOptions, SchemaOptions } from 'yaml';

const ORDERED_DICT_TAG = 'tag:yaml.org,2002:python/object/apply:collections.OrderedDict';

export class JobConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'JobConfigError';
  }
}

export interface JobProcessConfig {
  sqlite_db_path?: string;
  [key: string]: unknown;
}

export interface JobConfigDocument {
  config: {
    process: JobProcessConfig[];
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isEnoent(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT';
}

function plainFromNode(node: unknown): unknown {
  if (node == null) {
    return null;
  }
  if (isScalar(node)) {
    return node.value;
  }
  if (isPair(node)) {
    return [plainFromNode(node.key), plainFromNode(node.value)];
  }
  if (isSeq(node)) {
    return node.items.map(item => plainFromNode(item));
  }
  if (isMap(node)) {
    const obj: Record<string, unknown> = {};
    for (const pair of node.items) {
      const key = plainFromNode(pair.key);
      if (typeof key !== 'string') {
        throw new JobConfigError('YAML map key must be a string');
      }
      obj[key] = plainFromNode(pair.value);
    }
    return obj;
  }
  throw new JobConfigError('Unsupported YAML node in config.yaml');
}

// PyYAML dumps OrderedDict as one sequence argument: a list of [key, value] pairs.
const orderedDictTag: CollectionTag = {
  collection: 'seq',
  tag: ORDERED_DICT_TAG,
  resolve(value) {
    if (!isSeq(value)) {
      throw new JobConfigError('OrderedDict tag must be a sequence');
    }
    const args = plainFromNode(value);
    if (!Array.isArray(args) || args.length !== 1 || !Array.isArray(args[0])) {
      throw new JobConfigError('OrderedDict payload must be a single list of pairs');
    }
    const obj: Record<string, unknown> = {};
    for (const pair of args[0]) {
      if (!Array.isArray(pair) || pair.length < 2 || typeof pair[0] !== 'string') {
        throw new JobConfigError('OrderedDict entry must be a string key and a value');
      }
      obj[pair[0]] = pair[1];
    }
    return obj;
  },
};

const parseOptions: ParseOptions & SchemaOptions = {
  customTags: [orderedDictTag],
};

export function assertJobConfigDocument(value: unknown): JobConfigDocument {
  if (!isRecord(value)) {
    throw new JobConfigError('Job config must be an object');
  }
  const config = value.config;
  if (!isRecord(config)) {
    throw new JobConfigError('Job config is missing config');
  }
  const process = config.process;
  if (!Array.isArray(process) || process.length === 0 || !isRecord(process[0])) {
    throw new JobConfigError('Job config is missing config.process[0]');
  }
  return value as JobConfigDocument;
}

export function parseJobConfigYaml(text: string): JobConfigDocument {
  let doc: Document;
  try {
    doc = YAML.parseDocument(text, parseOptions);
  } catch (error) {
    if (error instanceof JobConfigError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new JobConfigError(`Invalid config.yaml: ${message}`);
  }
  if (doc.errors.length > 0) {
    throw new JobConfigError(`Invalid config.yaml: ${doc.errors[0].message}`);
  }
  const unresolved = doc.warnings.find(
    warning => warning.code === 'TAG_RESOLVE_FAILED' || warning.code === 'BAD_COLLECTION_TYPE',
  );
  if (unresolved) {
    throw new JobConfigError(`Invalid config.yaml: ${unresolved.message}`);
  }
  return assertJobConfigDocument(doc.toJS());
}

function parseDbJobConfig(dbJobConfig: string): JobConfigDocument {
  let parsed: unknown;
  try {
    parsed = JSON.parse(dbJobConfig);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new JobConfigError(`Invalid job_config JSON: ${message}`);
  }
  return assertJobConfigDocument(parsed);
}

type YamlFileRead =
  | { status: 'missing' }
  | { status: 'invalid'; message: string }
  | { status: 'ok'; document: JobConfigDocument };

function readConfigYamlFile(trainingFolder: string): YamlFileRead {
  const yamlPath = path.join(trainingFolder, 'config.yaml');
  let stat: fs.Stats;
  try {
    stat = fs.statSync(yamlPath);
  } catch (error) {
    if (isEnoent(error)) {
      return { status: 'missing' };
    }
    throw error;
  }
  if (!stat.isFile()) {
    return { status: 'invalid', message: 'config.yaml is not a file' };
  }
  try {
    return { status: 'ok', document: parseJobConfigYaml(fs.readFileSync(yamlPath, 'utf8')) };
  } catch (error) {
    if (error instanceof JobConfigError) {
      return { status: 'invalid', message: error.message };
    }
    throw error;
  }
}

export interface LoadedJobConfig {
  source: 'yaml' | 'database';
  document: JobConfigDocument;
}

export function loadJobConfig(trainingFolder: string, dbJobConfig: string): LoadedJobConfig {
  const yamlFile = readConfigYamlFile(trainingFolder);
  if (yamlFile.status === 'missing') {
    return { source: 'database', document: parseDbJobConfig(dbJobConfig) };
  }
  if (yamlFile.status === 'invalid') {
    throw new JobConfigError(yamlFile.message);
  }
  return { source: 'yaml', document: yamlFile.document };
}

export function loadJobConfigDocument(trainingFolder: string, dbJobConfig: string): JobConfigDocument {
  return loadJobConfig(trainingFolder, dbJobConfig).document;
}

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(item => canonicalize(item));
  }
  if (!isRecord(value)) {
    return value;
  }
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(value).sort()) {
    sorted[key] = canonicalize(value[key]);
  }
  return sorted;
}

export function jobConfigDocumentsEqual(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonicalize(left)) === JSON.stringify(canonicalize(right));
}

export function storedJobConfigMatches(stored: string, document: JobConfigDocument): boolean {
  if (stored === JSON.stringify(document)) {
    return true;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(stored);
  } catch (error) {
    if (error instanceof SyntaxError) {
      return false;
    }
    throw error;
  }
  return jobConfigDocumentsEqual(parsed, document);
}

export type JobConfigProjection =
  | { action: 'keep' }
  | { action: 'replace'; jobConfig: string }
  | { action: 'invalid'; message: string };

export function projectJobConfig(trainingFolder: string, dbJobConfig: string): JobConfigProjection {
  const yamlFile = readConfigYamlFile(trainingFolder);
  if (yamlFile.status === 'missing') {
    return { action: 'keep' };
  }
  if (yamlFile.status === 'invalid') {
    return { action: 'invalid', message: yamlFile.message };
  }
  if (storedJobConfigMatches(dbJobConfig, yamlFile.document)) {
    return { action: 'keep' };
  }
  return { action: 'replace', jobConfig: JSON.stringify(yamlFile.document) };
}

export function applySqliteDbPath(jobConfig: JobConfigDocument, sqliteDbPath: string): void {
  jobConfig.config.process[0].sqlite_db_path = sqliteDbPath;
}

export function launchJobConfig(document: JobConfigDocument, sqliteDbPath: string): JobConfigDocument {
  const launch = JSON.parse(JSON.stringify(document)) as JobConfigDocument;
  applySqliteDbPath(launch, sqliteDbPath);
  return launch;
}

export function writeJobConfigYamlIfPresent(
  trainingFolder: string,
  document: unknown,
): { wroteFile: boolean; jobConfig: string } {
  const jobConfig = JSON.stringify(document);
  const yamlPath = path.join(trainingFolder, 'config.yaml');
  let stat: fs.Stats;
  try {
    stat = fs.statSync(yamlPath);
  } catch (error) {
    if (isEnoent(error)) {
      return { wroteFile: false, jobConfig };
    }
    throw error;
  }
  if (!stat.isFile()) {
    throw new JobConfigError('config.yaml is not a file');
  }
  const yamlText = YAML.stringify(document);
  fs.writeFileSync(yamlPath, yamlText.endsWith('\n') ? yamlText : `${yamlText}\n`, 'utf8');
  return { wroteFile: true, jobConfig };
}
