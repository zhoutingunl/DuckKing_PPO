# 3D 官方网格坐标排错记录

## 已确认的根因

MuJoCo 的 `mesh_vert` 是共享网格顶点。对本项目使用的官方 XML，编译后的 `model.geom_pos` / `model.geom_quat` 已经包含 XML mesh 的局部偏移与旋转；运行时对应 `data.geom_xpos` / `data.geom_xmat`。浏览器中的最终顶点位置是：

```text
geom_world_pose × mesh_vert
```

`mesh_pos/mesh_quat` 不能在浏览器再次叠加。DuckKing 的已验证渲染器同样直接使用 geom 世界矩阵；PPO viewer 之前误把已编译的 mesh frame 再应用一次，导致头部、外壳和支架贴地或偏离身体。

## 已排除的错误做法

- 只给父组增加起点偏移，不能修复零件散开；它只能解决整只鸭的平移。
- 把 mesh 世界位置减去 root，再把 root 加回父组，会重复变换，造成远处白色碎片。
- 在已编译模型上再次应用 `mesh_pos/mesh_quat`，会得到“脑袋贴地、身体不完整”。

## 当前不变量

浏览器中每个策略鸭的层级为：

```text
world (MuJoCo x/y/z 映射到 Three x/y/z)
└── agent group (仅负责用户起点相对默认 root 的平移)
    └── geom frame (geom_xpos / geom_xmat；已包含编译后的 mesh frame)
        └── shared mesh vertices
```

修改模型导出或 Three.js 装配时，必须保持这条变换链；只修改相机、标签尺寸或父组坐标不能替代它。
