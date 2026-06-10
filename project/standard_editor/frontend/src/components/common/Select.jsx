import React from 'react'
import { ChevronDown } from 'lucide-react'

const Select = ({ 
  value, 
  onChange, 
  options = [], 
  placeholder = '선택하세요...',
  disabled = false,
  className = '',
  ...props 
}) => {
  const baseClass = 'form-select'
  const classes = `${baseClass} ${className}`.trim()

  return (
    <div style={{ position: 'relative' }}>
      <select
        className={classes}
        value={value || ''}
        onChange={onChange}
        disabled={disabled}
        style={{
          width: '100%',
          padding: '0.5rem 2.5rem 0.5rem 0.75rem',
          background: 'var(--bg-primary)',
          border: '1px solid var(--border-color)',
          borderRadius: '0.5rem',
          color: 'var(--text-primary)',
          fontSize: '0.875rem',
          cursor: disabled ? 'not-allowed' : 'pointer',
          appearance: 'none',
          WebkitAppearance: 'none',
          MozAppearance: 'none',
        }}
        {...props}
      >
        {placeholder && (
          <option value="" disabled>
            {placeholder}
          </option>
        )}
        {options.map((option) => {
          const optionValue = typeof option === 'string' ? option : option.value
          const optionLabel = typeof option === 'string' ? option : option.label || option.value
          return (
            <option key={optionValue} value={optionValue}>
              {optionLabel}
            </option>
          )
        })}
      </select>
      <ChevronDown
        size={16}
        style={{
          position: 'absolute',
          right: '0.75rem',
          top: '50%',
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
          color: 'var(--text-secondary)',
        }}
      />
    </div>
  )
}

export default Select

