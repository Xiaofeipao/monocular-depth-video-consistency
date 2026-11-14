# 工作站通过7897代理更新GitHub分支

本指南发布当前 **Part 1/2、Direction B、C v1和C v2**。所有操作都在工作站完成，不再复制到本地电脑。流程为：筛选导出 → 独立克隆目标分支 → 完整替换文件树 → 校验/审阅 → 提交和推送。

**目标分支的最新文件树会被本次上传包完整替换**：旧分支独有的文件和隐藏文件也会删除。保留提交历史，采用普通push，不强制推送；其他分支不变。不要在实验项目目录执行清空或`git add .`。

默认仍使用之前的仓库/分支；如果目标改变，只修改下面的变量：

- 仓库：`https://github.com/Xiaofeipao/monocular-depth-video-consistency.git`
- 分支：`JiongxuChen`
- 工作站本机HTTP代理：`http://127.0.0.1:7897`

2026-11-14已只读验证：此HTTP代理可访问GitHub，目标分支存在。这仅确认联网和读取，不能代替账号的推送权限。本文准备了上传包与命令，尚未替用户提交或推送远端。

## 上传与保留范围

| 内容 | 处理 |
|---|---|
| Part 1A fixed基线、误差—覆盖率研究；Part 1B指定视频与NYUv2扩展 | 上传项目自有实现/配置/测试/说明 |
| Part 2A两模型原生米制及尺度/边界分析；Part 2B VDA与逐帧比较 | 上传项目自有实现/配置/测试/说明 |
| B三尺度规则、C v1联合滤波/消融/经典对照、C v2开发/冻结/七方法评价 | **全部上传** |
| `src/refinement`、`part3_studies`、`part3_c_v2`及相应pipelines/configs/scripts | **全部保留，不能只复制某一个目录** |
| 环境依赖、固定版本、测试、科学协议、中文README/PLAN/各Step/implemented | 上传 |
| 当前课程PDF、`Project3_CN_final.md`、来源记录及MIT许可 | 上传 |
| 文档里的指标表、结果分析和失败说明 | 上传文本，原数值保持 |
| `data/`，包括指定片段、图像、抽帧、清单和下载压缩包 | **整个目录不上传** |
| `checkpoints/`，包括所有权重和缓存 | **整个目录不上传** |
| `outputs/`，包括预测、视频、图片、CSV、冻结参数、日志和自动元数据 | **整个目录不上传** |
| 官方第三方仓库、SAM2、`github_repository/`、虚拟环境和Python缓存 | **不上传** |
| 旧课程PDF/旧译文、旧上传指南、`docs/figures/`中的生成图片和个人备忘录 | 不加入本次包 |

GitHub提供代码、配置、测试与说明，**不包含实验资源或独立结果文件**。B和C v2的冻结JSON属于outputs，不随包上传；复现时按Step从开发集重新生成。文档指向data/checkpoints/outputs/github_repository的路径需要本地相应资源才能打开；分支中不提供这些文件。指标表仍可直接阅读，未把结果删掉或假称已重新运行。

1B/2B当前70真实帧与课程100帧素材问题、C v2的性能取舍、Demo/最终报告门禁都如实保留。`implemented.md`保留执行历史，可能含已经清理的旧路径，当前操作以README及各Step为准。

## 1. 设置路径、分支和代理

使用工作站Bash，在**同一个终端**依次执行全文命令。这里只需要系统Git、curl与Python 3标准库，不必激活模型环境，也不会往base安装包。

```bash
cd /home/user/cjx/3201_Project
export PROJECT_SOURCE="$PWD"
export REPO_URL='https://github.com/Xiaofeipao/monocular-depth-video-consistency.git'
export TARGET_BRANCH='JiongxuChen'
export GITHUB_PROXY='http://127.0.0.1:7897'

# 仅本终端函数中的Git调用使用代理，不修改全局Git/模型下载配置。
git_net() {
  NO_PROXY= no_proxy= command git -c "http.proxy=$GITHUB_PROXY" "$@"
}

git --version
python3 --version
curl --proxy "$GITHUB_PROXY" --noproxy '' --connect-timeout 5 --max-time 20 \
  --silent --show-error --output /dev/null --write-out 'GitHub HTTP %{http_code}\n' \
  https://github.com/
git_net ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$TARGET_BRANCH"
```

curl应显示成功HTTP状态，最后一条必须返回目标分支SHA及`refs/heads/JiongxuChen`。失败时先处理端口、地址或认证，不继续清空步骤。`127.0.0.1`指工作站自身：7897必须在工作站可用，不是把GitHub地址改成7897。

采用HTTPS仓库地址，代理通过Git的`http.proxy`按次传入；不要使用SSH的`git@github.com:...`地址套用本流程。[Git代理配置文档](https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxy)

若Git提示认证，使用自己的GitHub账号与有该仓库写权限的凭据；HTTPS的Password提示应输入访问令牌而不是账户密码。已有凭据管理器可继续使用。不要把令牌写到URL、脚本或项目文件中。[GitHub HTTPS认证说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line)

## 2. 在项目外生成本次上传包

```bash
export PUBLISH_WORK="$(mktemp -d "$HOME/aiaa3201-publish-XXXXXXXX")"
export PACKAGE_DIR="$PUBLISH_WORK/project_upload"
python3 scripts/publishing/export_project.py --output "$PACKAGE_DIR"
printf '本次发布工作目录：%s\n' "$PUBLISH_WORK"
du -sh "$PACKAGE_DIR"
find "$PACKAGE_DIR" -type f | sort
```

导出脚本按白名单复制项目自有文件，检查B、C v1/v2及许可是否齐全，拒绝符号链接、疑似凭据文件与非预期大文件；不会遍历数据、权重、产物或第三方仓库。自己新增的配置仍应检查，确认没有私有令牌或密码。

会生成：

```text
$PUBLISH_WORK/
  project_upload/              # 本次准备提交的完整文件树
    PROJECT_EXPORT.json        # 文件清单与逐文件SHA256
    PUBLISHING.md              # 发布范围和恢复资源/复现顺序
    .gitignore
    .gitattributes             # 保持原始字节，不做行尾转换
    ...
  project_upload.tar.gz        # 本次包的本地备份，无需搬到其他电脑
  project_upload.tar.gz.sha256
```

此时还没有操作GitHub分支。请检查本次包，特别是`docs/stepB.md`、`docs/stepC.md`、`docs/stepC_v2.md`和三个实现目录都在；整个data/checkpoints/outputs均应不存在。每次发布都重新导出，不复用旧包。

## 3. 全新克隆目标分支并保存旧提交号

```bash
export BRANCH_DIR="$PUBLISH_WORK/repository"
git_net clone --single-branch --branch "$TARGET_BRANCH" "$REPO_URL" "$BRANCH_DIR"
export EXPECTED_HEAD="$(git -C "$BRANCH_DIR" rev-parse HEAD)"
printf '%s\n' "$EXPECTED_HEAD" > "$PUBLISH_WORK/before-upload-commit.txt"
git -C "$BRANCH_DIR" remote -v
git -C "$BRANCH_DIR" branch --show-current
git -C "$BRANCH_DIR" status --short
```

核对远端地址、分支名和干净工作区。旧提交号留在仓库外，可据此查看旧版。**BRANCH_DIR必须是刚才全新克隆的目录，不能改成当前实验项目，也不要使用有队友未提交修改的现有克隆。**

## 4. 完整替换、校验并暂存

**下面会删除独立克隆中除`.git/`外的全部旧文件，再复制本次包。** 新文件树中不再保留旧版独有文件，Git历史仍在。

```bash
python3 "$PROJECT_SOURCE/scripts/publishing/replace_branch.py" \
  --package "$PACKAGE_DIR" \
  --repository "$BRANCH_DIR" \
  --branch "$TARGET_BRANCH" \
  --expected-head "$EXPECTED_HEAD"
```

脚本先检查目录互相独立、目标分支和HEAD一致、无未提交/未跟踪/忽略文件，再验证上传包文件清单和SHA。通过后才替换，用`git add -A`同时暂存新增、修改、删除，设置`.sh`可执行位，并逐文件核对Git暂存字节。

必须看到`committed:false`、`pushed:false`及暂存文件数才继续。脚本本身不提交、不推送；失败时不要提交，可以保留包并重新克隆到另一个新目录排查。不要删除`.git/`或清空实验项目。

## 5. 审阅差异

```bash
git -C "$BRANCH_DIR" --no-pager diff --cached --stat
git -C "$BRANCH_DIR" --no-pager diff --cached --name-status
# 以下应无输出。
git -C "$BRANCH_DIR" ls-files data checkpoints outputs github_repository \
  Depth-Anything-V2 Video-Depth-Anything UniDepth ml-depth-pro SAM2
# 以下应包含B、C v1、C v2及完整来源许可。
git -C "$BRANCH_DIR" ls-files src/refinement src/part3_studies src/part3_c_v2 \
  pipelines/part3 pipelines/part3_studies pipelines/part3_c_v2 \
  configs/part3 configs/part3_studies configs/part3_c_v2 \
  scripts/part3 scripts/part3_studies scripts/part3_c_v2
```

`D`表示旧文件会从分支最新版本删除，符合完整替换目标。确认README、Step 1–4、Step B/C/C v2、环境和测试齐全，且没有视频、预测或凭据。若分页器出现`(END)`，按`q`退出。

## 6. 提交并通过7897推送

```bash
git -C "$BRANCH_DIR" commit -m 'Publish Parts 1 and 2, Direction B, and C v1/v2'
git_net -C "$BRANCH_DIR" push origin "HEAD:refs/heads/$TARGET_BRANCH"
```

若提示缺少姓名/邮箱，只为这个克隆配置自己的`user.name`、`user.email`再提交。若没有改动，内容已一致，无需空提交。

若push提示non-fast-forward，说明克隆后队友又提交了内容：保留包，在新的目录重新克隆，先看新提交，再重新替换和审阅。不要`--force`覆盖未查看的工作，也不要为了推送而直接清空远端历史。

## 7. 确认远端文件树

```bash
git -C "$BRANCH_DIR" rev-parse HEAD
git_net -C "$BRANCH_DIR" ls-remote --heads origin "refs/heads/$TARGET_BRANCH"
git -C "$BRANCH_DIR" status --short
```

本地HEAD与远端目标分支SHA应相同，工作区应干净。网页切换到目标分支，检查全部新内容齐全、旧版独有文件已删除、资源/视频/产物目录不存在；main及其他分支不应变化。

这次替换只影响分支最新文件树。如果以前提交过大文件，它们仍可能在Git历史中；本指南不执行历史重写。

## 别人克隆后的复现顺序

先阅读README与`PUBLISHING.md`，按`configs/versions.yaml`和Step 0取得四个固定上游版本，然后创建两个Conda环境。主环境依赖`-e ./ml-depth-pro`，必须先有上游目录，再创建环境。

另行准备数据/权重和指定视频，按Step 1–4恢复原生预测及评价。B复用795张开发原生预测并拟合开发尺度；C v1按固定设置运行，C v2按开发场景划分选参/冻结后评价654图。原始预测不替换，alignment:none与诊断轨道不混淆。未恢复outputs时不能直接验证那些本地正式结果；正式训练和长时间预测/评价仍由用户启动。
