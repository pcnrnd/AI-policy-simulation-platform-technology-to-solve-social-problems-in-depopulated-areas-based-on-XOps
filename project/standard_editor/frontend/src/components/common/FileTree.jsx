import React, { useState } from 'react'
import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from 'lucide-react'

const FileTreeItem = ({ node, level = 0, selectedFiles, onToggleFile, onToggleFolder }) => {
  const [isExpanded, setIsExpanded] = useState(level === 0) // 최상위는 기본적으로 열림
  const isFile = node.type === 'file'
  const isSelected = selectedFiles.includes(node.path)
  const hasChildren = node.children && node.children.length > 0

  const handleToggle = () => {
    if (isFile) {
      onToggleFile(node.path)
    } else if (hasChildren) {
      setIsExpanded(!isExpanded)
      onToggleFolder(node.path)
    }
  }

  const handleCheckboxChange = (e) => {
    e.stopPropagation()
    if (isFile) {
      onToggleFile(node.path)
    }
  }

  const paddingLeft = level * 20 + 8

  return (
    <div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 0',
          paddingLeft: `${paddingLeft}px`,
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={handleToggle}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent'
        }}
      >
        {!isFile && hasChildren && (
          <span style={{ marginRight: '4px', display: 'inline-flex', alignItems: 'center' }}>
            {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>
        )}
        {!isFile && !hasChildren && <span style={{ width: '20px', display: 'inline-block' }} />}
        
        {isFile && (
          <input
            type="checkbox"
            checked={isSelected}
            onChange={handleCheckboxChange}
            onClick={(e) => e.stopPropagation()}
            style={{ marginRight: '8px' }}
          />
        )}
        {!isFile && (
          <span style={{ width: '20px', display: 'inline-block', marginRight: '4px' }} />
        )}

        <span style={{ marginRight: '8px', display: 'inline-flex', alignItems: 'center' }}>
          {isFile ? (
            <File size={16} />
          ) : isExpanded ? (
            <FolderOpen size={16} />
          ) : (
            <Folder size={16} />
          )}
        </span>
        <span>{node.name}</span>
      </div>

      {!isFile && hasChildren && isExpanded && (
        <div>
          {node.children.map((child) => (
            <FileTreeItem
              key={child.path}
              node={child}
              level={level + 1}
              selectedFiles={selectedFiles}
              onToggleFile={onToggleFile}
              onToggleFolder={onToggleFolder}
            />
          ))}
        </div>
      )}
    </div>
  )
}

const FileTree = ({ tree, selectedFiles, onToggleFile }) => {
  const handleToggleFolder = (path) => {
    // 폴더 토글 로직 (필요시 구현)
  }

  if (!tree || tree.length === 0) {
    return (
      <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
        파일이 없습니다.
      </div>
    )
  }

  return (
    <div
      style={{
        border: '1px solid var(--border-color)',
        borderRadius: '0.5rem',
        padding: '0.5rem',
        maxHeight: '400px',
        overflowY: 'auto',
        background: 'var(--bg-primary)',
      }}
    >
      {tree.map((node) => (
        <FileTreeItem
          key={node.path}
          node={node}
          selectedFiles={selectedFiles}
          onToggleFile={onToggleFile}
          onToggleFolder={handleToggleFolder}
        />
      ))}
    </div>
  )
}

export default FileTree

