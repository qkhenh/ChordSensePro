import { useRef } from 'react'
import { motion, useScroll, useTransform } from 'motion/react'
import { ChevronDown, Zap, Brain, Sparkles, AudioWaveform, Globe, Mail, Mic, Music, MapPin, Phone, Send } from 'lucide-react'

/* ── Global keyframes injected once ───────────────────────────────────────── */
const ShineStyle = () => (
  <style>{`
    @keyframes shine {
      0%   { transform: translateX(-120%) skewX(-20deg); }
      100% { transform: translateX(350%)  skewX(-20deg); }
    }
    @keyframes textGlow {
      0%, 100% { text-shadow: 0 0 20px rgba(167,139,250,0.5), 0 0 60px rgba(6,182,212,0.2); }
      50%       { text-shadow: 0 0 40px rgba(167,139,250,0.9), 0 0 100px rgba(6,182,212,0.4); }
    }
    @keyframes float {
      0%, 100% { transform: translateY(0); }
      50%       { transform: translateY(-8px); }
    }
    .logo-wrap:hover .logo-shine { animation: shine 0.7s ease forwards; }
    .text-glow { animation: textGlow 3s ease-in-out infinite; }
  `}</style>
)

/* ── Logo icon — dùng draft_logo2.jpg ─────────────────────────────────────── */
const LogoIcon = ({ size = 48 }) => (
  <div
    className="logo-wrap"
    style={{
      width: size, height: size, borderRadius: size * 0.22,
      position: 'relative', overflow: 'hidden', flexShrink: 0,
      boxShadow: `0 0 ${size * 0.8}px rgba(124,58,237,0.6),
                  0 0 ${size * 0.3}px rgba(6,182,212,0.35),
                  0 0 0 1px rgba(167,139,250,0.3)`,
    }}
  >
    {/* Shine sweep on hover */}
    <div className="logo-shine" style={{
      position: 'absolute', top: 0, left: 0, width: '45%', height: '100%',
      background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent)',
      transform: 'translateX(-120%) skewX(-20deg)',
      pointerEvents: 'none', zIndex: 2,
    }} />
    <img
      src="/draft_logo2.jpg"
      alt="ChordSense Pro Logo"
      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
    />
  </div>
)

/* ── Feature pill ──────────────────────────────────────────────────────────── */
const FeaturePill = ({ icon: Icon, label, delay }) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '9px 20px', borderRadius: 999,
      background: 'rgba(255,255,255,0.06)',
      border: '1px solid rgba(255,255,255,0.14)',
      backdropFilter: 'blur(10px)',
      color: 'rgba(255,255,255,0.82)',
      fontSize: 13, fontWeight: 500, whiteSpace: 'nowrap',
      boxShadow: '0 1px 0 rgba(255,255,255,0.07) inset',
    }}
  >
    <Icon size={15} style={{ color: '#a78bfa' }} />
    {label}
  </motion.div>
)

/* ── Card link column ──────────────────────────────────────────────────────── */
const LinkColumn = ({ title, links }) => (
  <div>
    <h4 style={{ textTransform: 'uppercase', letterSpacing: '0.14em', fontSize: 11, fontWeight: 700, color: '#111', marginBottom: 16, margin: '0 0 16px' }}>
      {title}
    </h4>
    <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
      {links.map(link => (
        <li key={link}>
          <a href="#" style={{ color: '#6b7280', fontWeight: 500, fontSize: 14, textDecoration: 'none', transition: 'color 0.2s' }}
            onMouseEnter={e => e.target.style.color = '#7C3AED'}
            onMouseLeave={e => e.target.style.color = '#6b7280'}
          >{link}</a>
        </li>
      ))}
    </ul>
  </div>
)

/* ── App ───────────────────────────────────────────────────────────────────── */
export default function App() {
  const containerRef = useRef(null)
  const { scrollYProgress } = useScroll({ target: containerRef })
  const y = useTransform(scrollYProgress, [0, 1], [-60, 160])

  const socialIcons = [Mail, Music, Mic, Globe]

  return (
    <div style={{ fontFamily: "'Inter', sans-serif", background: '#f8f9fa', minHeight: '100vh' }}>
      <ShineStyle />

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 1. HERO SECTION                                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section style={{
        minHeight: '100vh',
        background: 'linear-gradient(160deg, #04040f 0%, #0b0320 50%, #060d20 100%)',
        position: 'relative', overflow: 'hidden',
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
        padding: '80px 24px',
      }}>
        {/* Glow orbs */}
        {[
          { top: '10%', left: '15%', w: 600, h: 600, color: 'rgba(109,40,217,0.22)' },
          { top: '55%', right: '10%', w: 450, h: 450, color: 'rgba(6,182,212,0.16)' },
          { bottom: '5%', left: '35%', w: 300, h: 300, color: 'rgba(167,139,250,0.1)' },
        ].map((o, i) => (
          <div key={i} style={{
            position: 'absolute', width: o.w, height: o.h, borderRadius: '50%',
            top: o.top, bottom: o.bottom, left: o.left, right: o.right,
            background: `radial-gradient(circle, ${o.color} 0%, transparent 70%)`,
            filter: 'blur(50px)', pointerEvents: 'none',
          }} />
        ))}

        {/* Grid */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          backgroundImage: 'linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)',
          backgroundSize: '64px 64px',
        }} />

        {/* Vignette */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none',
          background: 'radial-gradient(ellipse at center, transparent 30%, rgba(4,4,15,0.7) 100%)',
        }} />

        {/* Content */}
        <div style={{ position: 'relative', textAlign: 'center', maxWidth: 780, zIndex: 1 }}>

          {/* Logo animated float */}
          <motion.div
            initial={{ opacity: 0, scale: 0.75 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, ease: [0.34, 1.56, 0.64, 1] }}
            style={{ display: 'flex', justifyContent: 'center', marginBottom: 32, animation: 'float 4s ease-in-out infinite' }}
          >
            <LogoIcon size={80} />
          </motion.div>

          {/* Title with glow */}
          <motion.h1
            initial={{ opacity: 0, y: 28 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.18 }}
            className="text-glow"
            style={{
              fontSize: 'clamp(44px, 7.5vw, 80px)',
              fontWeight: 800, letterSpacing: '-0.04em',
              color: '#fff', margin: '0 0 18px', lineHeight: 1.05,
              textShadow: '0 0 30px rgba(167,139,250,0.6), 0 0 80px rgba(6,182,212,0.25)',
            }}
          >
            ChordSense
            <span style={{
              background: 'linear-gradient(90deg, #c4b5fd, #67e8f9, #a78bfa)',
              backgroundSize: '200% auto',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              filter: 'drop-shadow(0 0 12px rgba(167,139,250,0.7))',
            }}> Pro</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            style={{
              fontSize: 'clamp(14px, 1.8vw, 18px)',
              color: 'rgba(255,255,255,0.5)', fontWeight: 400,
              margin: '0 0 44px', lineHeight: 1.65,
            }}
          >
            An AI-Powered Context-Aware System for Musician's<br />
            Interactive Practice Workspace with Harmonic Analysis Support
          </motion.p>

          {/* Pills */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginBottom: 64 }}>
            <FeaturePill icon={Brain}         label="MERT-v1-330M AI"           delay={0.42} />
            <FeaturePill icon={AudioWaveform} label="Extended Chord Recognition" delay={0.50} />
            <FeaturePill icon={Zap}           label="Real-time Practice"         delay={0.58} />
            <FeaturePill icon={Sparkles}      label="Harmonic Analysis"          delay={0.66} />
          </div>

          {/* Scroll indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            onClick={() => window.scrollTo({ top: window.innerHeight, behavior: 'smooth' })}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, color: 'rgba(255,255,255,0.28)', cursor: 'pointer' }}
          >
            <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.22em', textTransform: 'uppercase' }}>
              Scroll to explore
            </span>
            <motion.div animate={{ y: [0, 7, 0] }} transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}>
              <ChevronDown size={20} />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 2. PARALLAX SECTION                                                */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <section
        ref={containerRef}
        style={{
          position: 'relative', height: '100vh', overflow: 'hidden',
          background: '#070418',
        }}
      >
        <img
          src="https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=1920&q=80"
          alt=""
          style={{
            position: 'absolute', inset: 0, width: '100%', height: '100%',
            objectFit: 'cover', objectPosition: 'center',
            zIndex: 0,
          }}
        />
        {/* Cool purple overlay */}
        <div style={{
          position: 'absolute', inset: 0, zIndex: 1,
          background: 'linear-gradient(160deg, rgba(10,4,35,0.65) 0%, rgba(4,10,35,0.45) 50%, rgba(8,4,25,0.75) 100%)',
        }} />

        {/* Card */}
        <div style={{ position: 'absolute', top: 0, width: '100%', padding: '48px 24px 0', zIndex: 10 }}>
          <div style={{ maxWidth: 1280, margin: '0 auto' }}>
            <motion.div
              initial={{ opacity: 0, y: -24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              style={{
                background: 'rgba(255,255,255,0.97)',
                backdropFilter: 'blur(16px)',
                borderRadius: 24,
                boxShadow: '0 24px 64px rgba(0,0,0,0.3), 0 1px 0 rgba(255,255,255,0.8) inset',
                overflow: 'hidden',
              }}
            >
              {/* Top */}
              <div style={{
                padding: '32px 40px',
                display: 'flex', flexWrap: 'wrap',
                justifyContent: 'space-between', alignItems: 'flex-start', gap: 32,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  <LogoIcon size={48} />
                  <div>
                    <div style={{ fontSize: 26, fontWeight: 700, letterSpacing: '-0.03em', color: '#111', lineHeight: 1 }}>
                      ChordSense
                      <span style={{ background: 'linear-gradient(90deg,#7C3AED,#06B6D4)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Pro</span>
                    </div>
                    <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 5, fontWeight: 500 }}>
                      AI-Powered Music Practice Workspace
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 48, flexWrap: 'wrap' }}>
                  <LinkColumn title="Product"  links={['AI Chord Engine', 'Smart Grid View', 'Harmonic Analysis', 'Chord Dictionary']} />
                  <LinkColumn title="Research" links={['VN Piano Dataset', 'MERT-v1-330M', 'Extended Chords']} />
                  <LinkColumn title="Legal"    links={['Privacy Policy', 'Terms of Use']} />
                </div>
              </div>
              {/* Bottom bar */}
              <div style={{
                borderTop: '1px solid #f3f4f6', background: '#fff',
                padding: '16px 40px',
                display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16,
              }}>
                <p style={{ fontSize: 13, color: '#9ca3af', fontWeight: 500, margin: 0 }}>
                  © 2026 ChordSense Pro. All Rights Reserved.
                </p>
                <div style={{ display: 'flex', gap: 8 }}>
                  {socialIcons.map((Icon, i) => (
                    <a key={i} href="#" style={{
                      width: 40, height: 40, borderRadius: '50%',
                      border: '1px solid #f3f4f6',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: '#9ca3af', textDecoration: 'none', transition: 'all 0.25s',
                    }}
                      onMouseEnter={e => { e.currentTarget.style.background = '#7C3AED'; e.currentTarget.style.borderColor = '#7C3AED'; e.currentTarget.style.color = '#fff' }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = '#f3f4f6'; e.currentTarget.style.color = '#9ca3af' }}
                    >
                      <Icon size={18} />
                    </a>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Parallax foreground — piano keys, no faces */}
        <motion.div style={{ y, position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 5 }}>
          <img
            src="https://images.unsplash.com/photo-1552422535-c45813c61732?w=1280&q=85"
            alt="piano keys"
            style={{
              width: '100%', height: '100%',
              objectFit: 'cover', objectPosition: 'center bottom',
              opacity: 0.3,
              filter: 'hue-rotate(200deg) saturate(0.6) brightness(0.85)',
              maskImage: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 60%)',
              WebkitMaskImage: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0) 60%)',
            }}
          />
        </motion.div>
      </section>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* 3. CONTACT / FOOTER SECTION                                        */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <footer style={{
        background: 'linear-gradient(160deg, #04040f 0%, #0b0320 60%, #060d20 100%)',
        position: 'relative', overflow: 'hidden', padding: '96px 24px 0',
      }}>
        {/* Glow */}
        <div style={{
          position: 'absolute', top: '-20%', left: '50%', transform: 'translateX(-50%)',
          width: 700, height: 400, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%)',
          filter: 'blur(60px)', pointerEvents: 'none',
        }} />

        <div style={{ maxWidth: 1100, margin: '0 auto', position: 'relative', zIndex: 1 }}>

          {/* Top — logo + tagline */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            style={{ textAlign: 'center', marginBottom: 72 }}
          >
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
              <LogoIcon size={56} />
            </div>
            <h2 style={{
              fontSize: 'clamp(28px, 4vw, 44px)', fontWeight: 800,
              letterSpacing: '-0.03em', color: '#fff', margin: '0 0 12px',
              textShadow: '0 0 40px rgba(167,139,250,0.4)',
            }}>
              ChordSense<span style={{
                background: 'linear-gradient(90deg,#c4b5fd,#67e8f9)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}> Pro</span>
            </h2>
            <p style={{ color: 'rgba(255,255,255,0.4)', fontSize: 15, fontWeight: 400, margin: 0, fontStyle: 'italic' }}>
              Phân tích sâu — Luyện tập chuẩn — Thành thạo toàn diện
            </p>
          </motion.div>

          {/* Contact cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: 20, marginBottom: 72,
            }}
          >
            {[
              {
                icon: Mail, title: 'Email',
                lines: ['khanh.dq@hcmut.edu.vn', 'tien.mc@hcmut.edu.vn'],
                color: '#a78bfa',
              },
              {
                icon: MapPin, title: 'Location',
                lines: ['Faculty of Computer Science', 'Ho Chi Minh University of Technology'],
                color: '#67e8f9',
              },
              {
                icon: Send, title: 'Project',
                lines: ['IoT Multidisciplinary Project', 'Academic Year 2025 – 2026'],
                color: '#86efac',
              },
            ].map(({ icon: Icon, title, lines, color }) => (
              <div key={title} style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 16, padding: '28px 28px',
                backdropFilter: 'blur(8px)',
              }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 12, marginBottom: 16,
                  background: `${color}18`,
                  border: `1px solid ${color}30`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Icon size={20} style={{ color }} />
                </div>
                <h4 style={{ color: '#fff', fontWeight: 700, fontSize: 15, margin: '0 0 10px', letterSpacing: '-0.01em' }}>{title}</h4>
                {lines.map(l => (
                  <p key={l} style={{ color: 'rgba(255,255,255,0.45)', fontSize: 13, margin: '0 0 4px', fontWeight: 400 }}>{l}</p>
                ))}
              </div>
            ))}
          </motion.div>

          {/* Team */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            style={{ marginBottom: 64 }}
          >
            <h3 style={{ color: 'rgba(255,255,255,0.6)', fontSize: 11, fontWeight: 700, letterSpacing: '0.2em', textTransform: 'uppercase', textAlign: 'center', marginBottom: 24 }}>
              Development Team — CC01, Team 05
            </h3>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 20, flexWrap: 'wrap' }}>
              {[
                { name: 'Đặng Quốc Khánh', id: '2352515', role: 'Data Engineer & AI Developer', badge: '#7C3AED' },
                { name: 'Mai Chung Tiến',  id: '2353177', role: 'Full Stack Developer',          badge: '#06B6D4' },
              ].map(m => (
                <div key={m.id} style={{
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  borderRadius: 14, padding: '20px 28px',
                  minWidth: 240, textAlign: 'center',
                }}>
                  <div style={{
                    display: 'inline-block', padding: '3px 10px', borderRadius: 999,
                    background: `${m.badge}20`, border: `1px solid ${m.badge}40`,
                    color: m.badge, fontSize: 11, fontWeight: 600, marginBottom: 10,
                  }}>{m.id}</div>
                  <div style={{ color: '#fff', fontWeight: 700, fontSize: 15, marginBottom: 4 }}>{m.name}</div>
                  <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 12 }}>{m.role}</div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Copyright bar */}
        <div style={{
          borderTop: '1px solid rgba(255,255,255,0.07)',
          padding: '20px 24px',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <p style={{ color: 'rgba(255,255,255,0.22)', fontSize: 12, margin: 0, fontWeight: 500 }}>
            © 2026 ChordSense Pro · HCMUT Faculty of Computer Science and Engineering
          </p>
        </div>
      </footer>
    </div>
  )
}
