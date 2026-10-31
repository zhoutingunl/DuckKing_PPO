# 3D 官方网格坐标排错记录

## 已确认的根因

MuJoCo 的 `mesh_vert` 是共享网格坐标，不能直接当作 geom 坐标使用。每个视觉 geom 的最终顶点位置是：

```text
geom_world_pose × mesh_local_pose(mesh_pos, mesh_quat) × mesh_vert
```

其中 `geom_xpos/geom_xmat` 只提供 geom frame 的世界位姿；`mesh_pos/mesh_quat` 还必须叠加。DuckKing 的已验证渲染器通过 body/geom 层级保留了这次变换，PPO viewer 之前把 mesh 顶点直接挂在 geom 世界位姿下，导致外壳、支架、轴承等零件散开。

## 已排除的错误做法

- 只给父组增加起点偏移，不能修复零件散开；它只能解决整只鸭的平移。
- 把 mesh 世界位置减去 root，再把 root 加回父组，会重复变换，造成远处白色碎片。
- 把 `geom_xpos` 当成 mesh 顶点坐标，忽略 `mesh_pos/mesh_quat`，会得到截图中的“中心小块 + 远处小块”。

## 当前不变量

浏览器中每个策略鸭的层级为：

```text
world (MuJoCo x/y/z 映射到 Three x/y/z)
└── agent group (仅负责用户起点相对默认 root 的平移)
    └── geom frame (geom_xpos / geom_xmat)
        └── mesh frame (mesh_pos / mesh_quat)
            └── shared mesh vertices
```

修改模型导出或 Three.js 装配时，必须保持这条变换链；只修改相机、标签尺寸或父组坐标不能替代它。
