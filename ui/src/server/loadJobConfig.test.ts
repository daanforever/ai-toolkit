import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  applySqliteDbPath,
  launchJobConfig,
  loadJobConfig,
  loadJobConfigDocument,
  parseJobConfigYaml,
  projectJobConfig,
  writeJobConfigYamlIfPresent,
} from '../../cron/loadJobConfig';

const roots: string[] = [];

const document = {
  job: 'extension',
  config: {
    name: 'sample_job',
    process: [
      {
        type: 'diffusion_trainer',
        flag: true,
        note: null,
        lr: 0.0001,
        network: { linear: 32, conv: 16 },
      },
    ],
  },
};

const dbJobConfig = JSON.stringify(document);

const sentinelDbJobConfig = JSON.stringify({
  job: 'from-db',
  config: { process: [{ type: 'from-db' }] },
});

// Copied from PyYAML 6 yaml.dump of an OrderedDict with the same contents as `document`.
const orderedDictYaml = `!!python/object/apply:collections.OrderedDict
- - - job
    - extension
  - - config
    - !!python/object/apply:collections.OrderedDict
      - - - name
          - sample_job
        - - process
          - - !!python/object/apply:collections.OrderedDict
              - - - type
                  - diffusion_trainer
                - - flag
                  - true
                - - note
                  - null
                - - lr
                  - 0.0001
                - - network
                  - !!python/object/apply:collections.OrderedDict
                    - - - linear
                        - 32
                      - - conv
                        - 16
`;

const plainYaml = `job: extension
config:
  name: sample_job
  process:
    - type: diffusion_trainer
      flag: true
      note: null
      lr: 0.0001
      network:
        linear: 32
        conv: 16
`;

const sqliteDbPath = path.join('toolkit-root', 'aitk_db.db');

function makeRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'job-config-'));
  roots.push(root);
  return root;
}

function writeConfig(root: string, contents: string): void {
  fs.writeFileSync(path.join(root, 'config.yaml'), contents);
}

afterEach(() => {
  while (roots.length > 0) {
    const root = roots.pop();
    if (root) fs.rmSync(root, { recursive: true, force: true });
  }
});

describe('loadJobConfigDocument', () => {
  it('uses the database json when config.yaml is missing', () => {
    const root = makeRoot();
    expect(loadJobConfigDocument(root, dbJobConfig)).toEqual(document);
  });

  it('loads a PyYAML OrderedDict dump instead of the database', () => {
    const root = makeRoot();
    writeConfig(root, orderedDictYaml);
    expect(loadJobConfigDocument(root, sentinelDbJobConfig)).toEqual(document);
  });

  it('loads plain yaml instead of the database', () => {
    const root = makeRoot();
    writeConfig(root, plainYaml);
    expect(loadJobConfigDocument(root, sentinelDbJobConfig)).toEqual(document);
  });

  it('rejects an empty file without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Job config must be an object/);
  });

  it('rejects a yaml list without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '- a\n- b\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Job config must be an object/);
  });

  it('rejects a mapping without config.process without using the database', () => {
    const root = makeRoot();
    writeConfig(root, 'job: extension\nconfig:\n  name: sample_job\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/config\.process\[0\]/);
  });

  it('rejects an unknown tag without using the database', () => {
    const root = makeRoot();
    writeConfig(root, '!!python/object/apply:os.system\n- echo\n');
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/Invalid config\.yaml/);
  });

  it('rejects a directory named config.yaml without using the database', () => {
    const root = makeRoot();
    fs.mkdirSync(path.join(root, 'config.yaml'));
    expect(() => loadJobConfigDocument(root, sentinelDbJobConfig)).toThrow(/not a file/);
  });

  it('sets sqlite_db_path on the database source and the yaml source', () => {
    const root = makeRoot();
    const fromDb = loadJobConfigDocument(root, dbJobConfig);
    applySqliteDbPath(fromDb, sqliteDbPath);
    expect(fromDb.config.process[0].sqlite_db_path).toBe(sqliteDbPath);

    writeConfig(root, orderedDictYaml);
    const fromYaml = loadJobConfigDocument(root, sentinelDbJobConfig);
    applySqliteDbPath(fromYaml, sqliteDbPath);
    expect(fromYaml.config.process[0].sqlite_db_path).toBe(sqliteDbPath);
    expect(fromYaml.config.process[0].lr).toBe(0.0001);
    expect(fromYaml.config.process[0].flag).toBe(true);
    expect(fromYaml.config.process[0].note).toBeNull();
  });
});

describe('config.yaml projection', () => {
  it('keeps the yaml document free of the launch sqlite path', () => {
    const root = makeRoot();
    writeConfig(root, orderedDictYaml);
    const loaded = loadJobConfig(root, sentinelDbJobConfig);
    expect(loaded.source).toBe('yaml');
    expect(loaded.document).toEqual(document);
    expect(loaded.document.config.process[0].sqlite_db_path).toBeUndefined();

    const launch = launchJobConfig(loaded.document, sqliteDbPath);
    expect(launch.config.process[0].sqlite_db_path).toBe(sqliteDbPath);
    expect(loaded.document.config.process[0].sqlite_db_path).toBeUndefined();

    const projection = projectJobConfig(root, sentinelDbJobConfig);
    expect(projection).toEqual({ action: 'replace', jobConfig: JSON.stringify(loaded.document) });
    if (projection.action === 'replace') {
      expect(JSON.parse(projection.jobConfig).config.process[0].sqlite_db_path).toBeUndefined();
    }
  });

  it('does not replace the column when config.yaml is missing', () => {
    const root = makeRoot();
    const loaded = loadJobConfig(root, dbJobConfig);
    expect(loaded.source).toBe('database');
    expect(projectJobConfig(root, dbJobConfig)).toEqual({ action: 'keep' });
  });

  it('does not request a write when the stored document already matches', () => {
    const root = makeRoot();
    writeConfig(root, plainYaml);
    const reordered = JSON.stringify({
      config: {
        process: document.config.process,
        name: 'sample_job',
      },
      job: 'extension',
    });
    expect(projectJobConfig(root, reordered)).toEqual({ action: 'keep' });

    const first = projectJobConfig(root, sentinelDbJobConfig);
    expect(first.action).toBe('replace');
    if (first.action !== 'replace') {
      throw new Error('expected a replacement projection');
    }
    expect(projectJobConfig(root, first.jobConfig)).toEqual({ action: 'keep' });
  });

  it('replaces the column when yaml lr differs', () => {
    const root = makeRoot();
    writeConfig(root, plainYaml);
    const stale = JSON.parse(dbJobConfig) as typeof document;
    stale.config.process[0].lr = 1;
    const projection = projectJobConfig(root, JSON.stringify(stale));
    expect(projection.action).toBe('replace');
    if (projection.action === 'replace') {
      expect(JSON.parse(projection.jobConfig)).toEqual(document);
    }
  });

  it('reports invalid yaml without a replacement document', () => {
    const root = makeRoot();
    writeConfig(root, '');
    const projection = projectJobConfig(root, sentinelDbJobConfig);
    expect(projection.action).toBe('invalid');
    expect(projection).not.toHaveProperty('jobConfig');
  });

  it('writes an existing config.yaml and stores the same json', () => {
    const root = makeRoot();
    writeConfig(root, plainYaml);
    const submitted = JSON.parse(dbJobConfig) as typeof document;
    submitted.config.process[0].lr = 0.0002;
    const saved = writeJobConfigYamlIfPresent(root, submitted);
    expect(saved.wroteFile).toBe(true);
    expect(JSON.parse(saved.jobConfig)).toEqual(submitted);
    const written = fs.readFileSync(path.join(root, 'config.yaml'), 'utf8');
    expect(parseJobConfigYaml(written)).toEqual(submitted);
    expect(projectJobConfig(root, saved.jobConfig)).toEqual({ action: 'keep' });
  });

  it('does not create config.yaml when saving without the file', () => {
    const root = makeRoot();
    const saved = writeJobConfigYamlIfPresent(root, document);
    expect(saved.wroteFile).toBe(false);
    expect(saved.jobConfig).toBe(JSON.stringify(document));
    expect(fs.existsSync(path.join(root, 'config.yaml'))).toBe(false);
  });
});
