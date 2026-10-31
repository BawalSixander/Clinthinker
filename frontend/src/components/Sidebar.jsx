import { CircleHelp, Database, FileQuestion, FileUp, NotebookText, Settings2, ShieldCheck } from "lucide-react";

const items = [
  ["Upload", FileUp],
  ["Generate Practice Questions", FileQuestion],
  ["Saved Questions", NotebookText],
  ["Training Data", Database],
  ["Settings", Settings2],
  ["Help", CircleHelp]
];

export default function Sidebar({ active, onNavigate, open, onClose }) {
  return (
    <>
      {open && <button className="drawer-scrim" aria-label="Close menu" onClick={onClose} />}
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><FileUp size={19} strokeWidth={1.8} /></div>
          <div className="brand-name">Clinthinker©</div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav">
          {items.map(([label, Icon]) => (
            <button key={label} className={`nav-item ${active === label ? "nav-active" : ""}`} onClick={() => onNavigate(label)}>
              <Icon size={18} strokeWidth={1.8} />
              <span>{label}</span>
              {active === label && <span className="nav-pip" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="privacy-note"><ShieldCheck size={15} /><span>Saved on this device</span></div>
          <div className="offline-sidebar"><span /> OFFLINE READY</div>
        </div>
      </aside>
    </>
  );
}
