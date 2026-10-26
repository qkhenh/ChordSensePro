import { useId, useState, type DragEvent } from 'react'
import { useI18n } from '../i18n/context'
import { ACCEPTED_EXTENSIONS, formatBytes, validateFile, type FileProblem } from './validate'

const ACCEPT = [...ACCEPTED_EXTENSIONS.map((e) => `.${e}`), 'audio/*'].join(',')

interface Props {
  onSubmit(file: File): void
  busy: boolean
}

/** Pick or drop one music file; it is checked here and only offered for analysis if usable. */
export function Dropzone({ onSubmit, busy }: Props) {
  const { t } = useI18n()
  const inputId = useId()
  const [file, setFile] = useState<File | null>(null)
  const [problem, setProblem] = useState<FileProblem | null>(null)
  const [over, setOver] = useState(false)

  function choose(next: File | undefined) {
    if (!next) return
    const found = validateFile(next)
    setProblem(found)
    setFile(found ? null : next)
  }

  function drop(event: DragEvent) {
    event.preventDefault()
    setOver(false)
    choose(event.dataTransfer.files[0])
  }

  return (
    <div>
      <div
        data-testid="dropzone"
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
        className={`rounded-xl border-2 border-dashed px-6 py-8 text-center transition ${
          over ? 'border-brand-purple-light bg-brand-purple/10' : 'border-white/15 bg-white/5'
        }`}
      >
        <label htmlFor={inputId} className="cursor-pointer text-slate-300">
          {t.analyze.dropPrompt}{' '}
          <span className="font-semibold text-brand-purple-light underline">{t.analyze.chooseFile}</span>
          <input
            id={inputId}
            type="file"
            accept={ACCEPT}
            className="sr-only"
            onChange={(e) => choose(e.target.files?.[0])}
          />
        </label>
        <p className="mt-2 text-sm text-slate-500">{t.analyze.formats}</p>
      </div>

      {problem && (
        <p role="alert" className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-300">
          {t.analyze.problems[problem]}
        </p>
      )}

      {file && (
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-lg bg-white/5 px-4 py-3">
          <p className="min-w-0 truncate text-slate-200">
            <span>{file.name}</span> <span className="text-sm text-slate-500">{formatBytes(file.size)}</span>
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => onSubmit(file)}
            className="rounded-lg bg-brand-purple px-4 py-2 font-semibold text-white transition hover:bg-brand-purple-light disabled:opacity-60"
          >
            {busy ? t.analyze.sending : t.analyze.submit}
          </button>
        </div>
      )}
    </div>
  )
}
