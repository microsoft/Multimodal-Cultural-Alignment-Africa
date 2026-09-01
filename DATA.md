# Data and privacy

This public repository contains only synthetic demonstration records. It does
not include the original annotation-platform exports, annotator identities,
study participant data, generated evaluation runs, or unpublished research
artifacts used during development.

To evaluate another dataset, convert each record to the schema demonstrated in
`examples/sample_stories.json`. Remove direct identifiers and confirm that you
have permission to process and publish any text, images, or derived results.
Every field under `person` is transmitted to the configured model provider and
stored in the output, so only include context necessary for the evaluation.

Remote image URLs are sent to the configured model provider. Use local image
files or an approved provider when the images are private or access-controlled.
Local filesystem paths are not included in model prompts.

The `StoryGen` component accepts profile CSVs
supplied by the operator. Profile fields are transmitted to the selected model
provider and copied into local task JSON for traceability. When image generation
is enabled, image prompts are processed by the configured local model. When
storage upload is enabled, generated images are sent to the configured
Supabase project.

Operators are responsible for de-identification, consent, institutional
approval, provider data-handling terms, access controls, retention, and deletion
of all non-synthetic inputs and outputs.

## Released judgments

`data/judgments/` contains de-identified human cultural-alignment annotations
and matching multi-agent LLM judgments for Hausa, isiXhosa, Kikuyu, Luo, and
Xichangana.

The release does not contain annotator names or email addresses, internal user
or submission IDs, raw story/study IDs, participant profile objects, HTML
profiles, prompts, search queries, or annotation timestamps. Story and
annotator IDs are replaced with language-local sequential identifiers.
Participant names and name components are replaced throughout the released
text. External media URLs are omitted because their object paths preserve raw
story identifiers. Source profile values such as age, location, religion,
occupation, dietary restrictions, disability, and health details are replaced
with `[REDACTED_PROFILE_ATTRIBUTE]`. Media counts and all image judgments remain
available.

See `data/judgments/README.md` and `data/judgments/manifest.json` for schemas,
counts, exclusions, and reproduction instructions.
