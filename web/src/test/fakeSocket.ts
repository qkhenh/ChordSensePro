/** A scriptable stand-in for WebSocket: tests decide when it opens, speaks and drops. */
export class FakeSocket {
  static instances: FakeSocket[] = []
  static failToConstruct = false

  url: string
  protocols: string[]
  closed = false
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: ((event: { code: number }) => void) | null = null
  onerror: (() => void) | null = null

  constructor(url: string, protocols: string[] = []) {
    if (FakeSocket.failToConstruct) throw new Error('cannot construct')
    this.url = url
    this.protocols = protocols
    FakeSocket.instances.push(this)
  }

  static get last(): FakeSocket {
    return FakeSocket.instances[FakeSocket.instances.length - 1]
  }

  static reset() {
    FakeSocket.instances = []
    FakeSocket.failToConstruct = false
  }

  close() {
    this.closed = true
  }

  open() {
    this.onopen?.()
  }

  send(data: unknown) {
    this.message(data)
  }

  message(data: unknown) {
    this.onmessage?.({ data: typeof data === 'string' ? data : JSON.stringify(data) })
  }

  /** The server (or network) ends the connection. */
  drop(code = 1006) {
    this.onclose?.({ code })
  }
}
