/**
 * 파일 트리 Mock Data 서비스
 * 저장소의 파일 목록을 트리 구조로 반환합니다.
 */

/**
 * Mock 파일 트리 데이터 생성
 * @param {string} repositoryPath - 저장소 경로
 * @returns {Promise<Array>} 파일 트리 데이터
 */
export const getFileTree = async (repositoryPath) => {
  // 실제로는 API를 통해 파일 목록을 가져오지만, 여기서는 Mock Data 반환
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve([
        {
          name: 'data',
          type: 'directory',
          path: 'data',
          children: [
            {
              name: 'raw',
              type: 'directory',
              path: 'data/raw',
              children: [
                {
                  name: 'restaurant_2020.csv',
                  type: 'file',
                  path: 'data/raw/restaurant_2020.csv',
                },
                {
                  name: 'restaurant_2021.csv',
                  type: 'file',
                  path: 'data/raw/restaurant_2021.csv',
                },
                {
                  name: 'restaurant_2022.csv',
                  type: 'file',
                  path: 'data/raw/restaurant_2022.csv',
                },
              ],
            },
            {
              name: 'processed',
              type: 'directory',
              path: 'data/processed',
              children: [
                {
                  name: 'merged_data.csv',
                  type: 'file',
                  path: 'data/processed/merged_data.csv',
                },
                {
                  name: 'cleaned_data.csv',
                  type: 'file',
                  path: 'data/processed/cleaned_data.csv',
                },
              ],
            },
          ],
        },
        {
          name: 'models',
          type: 'directory',
          path: 'models',
          children: [
            {
              name: 'model_v1.pkl',
              type: 'file',
              path: 'models/model_v1.pkl',
            },
            {
              name: 'model_v2.pkl',
              type: 'file',
              path: 'models/model_v2.pkl',
            },
          ],
        },
        {
          name: 'scripts',
          type: 'directory',
          path: 'scripts',
          children: [
            {
              name: 'preprocess.py',
              type: 'file',
              path: 'scripts/preprocess.py',
            },
            {
              name: 'train.py',
              type: 'file',
              path: 'scripts/train.py',
            },
            {
              name: 'utils',
              type: 'directory',
              path: 'scripts/utils',
              children: [
                {
                  name: 'helpers.py',
                  type: 'file',
                  path: 'scripts/utils/helpers.py',
                },
              ],
            },
          ],
        },
        {
          name: 'config.yaml',
          type: 'file',
          path: 'config.yaml',
        },
        {
          name: 'README.md',
          type: 'file',
          path: 'README.md',
        },
        {
          name: 'requirements.txt',
          type: 'file',
          path: 'requirements.txt',
        },
      ])
    }, 300) // API 호출 시뮬레이션
  })
}

/**
 * 파일 경로를 평면 배열로 변환 (체크박스 선택용)
 * @param {Array} tree - 파일 트리 데이터
 * @returns {Array} 평면화된 파일 경로 배열
 */
export const flattenFileTree = (tree) => {
  const files = []
  
  const traverse = (nodes) => {
    nodes.forEach((node) => {
      if (node.type === 'file') {
        files.push(node.path)
      } else if (node.children) {
        traverse(node.children)
      }
    })
  }
  
  traverse(tree)
  return files
}

